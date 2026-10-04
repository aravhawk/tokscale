//! Kilo CLI session parser
//!
//! Parses messages from the SQLite database under the XDG data dir
//! (`~/.local/share/kilo/kilo.db`, plus `kilo-<channel>.db` for non-stable
//! release channels).
//!
//! Current Kilo CLI writes one assistant step per `session_message` row
//! (role in the `type` column, model nested under `$.model`). Older builds
//! keep that payload in a `message` table with `$.role` and a top-level
//! `modelID`. Both are read through [`super::opencode_schema`]; the places
//! where Kilo departs from OpenCode are `OpenCodeSchemaConfig::kilo`.

use super::opencode_schema::{parse_opencode_schema_sqlite, OpenCodeSchemaConfig};
use super::utils::file_modified_timestamp_ms;
use super::UnifiedMessage;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub fn parse_kilo_sqlite(db_path: &Path) -> Vec<UnifiedMessage> {
    let fallback_timestamp = file_modified_timestamp_ms(db_path);
    parse_kilo_sqlite_with_fallback(db_path, fallback_timestamp)
}

/// Parse with an explicit timestamp for assistant turns whose payload carries
/// no `time` object. Kilo, unlike the other OpenCode-schema clients, keeps such
/// messages rather than dropping them.
pub fn parse_kilo_sqlite_with_fallback(
    db_path: &Path,
    fallback_timestamp: i64,
) -> Vec<UnifiedMessage> {
    parse_opencode_schema_sqlite(db_path, OpenCodeSchemaConfig::kilo(fallback_timestamp))
}

/// Parse every discovered Kilo database, collapsing a message that was copied
/// into more than one channel file.
///
/// Channel builds keep a separate `kilo-<channel>.db` next to `kilo.db`. The
/// same step can land in both when a user switches channels, and the row id
/// (the message id, since the payload omits `$.id`) is stable across that copy.
pub fn parse_kilo_databases(db_paths: &[PathBuf]) -> Vec<UnifiedMessage> {
    let mut seen = HashSet::new();
    let mut messages = Vec::new();
    for db_path in db_paths {
        for message in parse_kilo_sqlite(db_path) {
            if let Some(key) = message.dedup_key.as_deref() {
                if !seen.insert(key.to_string()) {
                    continue;
                }
            }
            messages.push(message);
        }
    }
    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sessions::opencode_schema::OpenCodeSchemaMessage;
    use rusqlite::{params, Connection};
    use tempfile::TempDir;

    fn create_kilo_sqlite_db(dir: &TempDir) -> std::path::PathBuf {
        let db_path = dir.path().join("kilo.db");
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE message (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                data TEXT NOT NULL
            );
            "#,
        )
        .unwrap();
        db_path
    }

    fn insert_kilo_message(conn: &Connection, row_id: &str, session_id: &str, data_json: &str) {
        conn.execute(
            "INSERT INTO message (id, session_id, data) VALUES (?1, ?2, ?3)",
            params![row_id, session_id, data_json],
        )
        .unwrap();
    }

    #[test]
    fn test_parse_kilo_message_structure() {
        let json = r#"{
            "id": "msg-123",
            "session_id": "sess-456",
            "role": "assistant",
            "modelID": "minimax/m2.5",
            "providerID": "kilo",
            "cost": 0.15,
            "tokens": {
                "input": 1000,
                "output": 200,
                "cache": {"read": 500, "write": 100}
            },
            "time": {"created": 1700000000000}
        }"#;

        let mut bytes = json.as_bytes().to_vec();
        let msg: OpenCodeSchemaMessage = simd_json::from_slice(&mut bytes).unwrap();
        assert_eq!(msg.role.as_deref(), Some("assistant"));
        assert_eq!(msg.cost, Some(0.15));
        assert_eq!(msg.model_id, Some("minimax/m2.5".to_string()));
    }

    #[test]
    fn test_parse_kilo_sqlite_reads_assistant_rows() {
        let dir = TempDir::new().unwrap();
        let db_path = create_kilo_sqlite_db(&dir);
        let conn = Connection::open(&db_path).unwrap();

        let data_json = r#"{
            "id": "embedded-msg-1",
            "session_id": "sess-1",
            "role": "assistant",
            "modelID": "claude-sonnet-4",
            "providerID": "anthropic",
            "cost": 0.42,
            "agent": "architect",
            "tokens": {
                "input": 1200,
                "output": 300,
                "reasoning": 40,
                "cache": {"read": 75, "write": 25}
            },
            "time": {"created": 1700000000123.0}
        }"#;
        insert_kilo_message(&conn, "row-msg-1", "sess-1", data_json);
        drop(conn);

        let messages = parse_kilo_sqlite_with_fallback(&db_path, 42);
        assert_eq!(messages.len(), 1);

        let msg = &messages[0];
        assert_eq!(msg.client, "kilo");
        assert_eq!(msg.session_id, "sess-1");
        assert_eq!(msg.model_id, "claude-sonnet-4");
        assert_eq!(msg.provider_id, "anthropic");
        assert_eq!(msg.timestamp, 1_700_000_000_123);
        assert_eq!(msg.tokens.input, 1200);
        assert_eq!(msg.tokens.output, 300);
        assert_eq!(msg.tokens.reasoning, 40);
        assert_eq!(msg.tokens.cache_read, 75);
        assert_eq!(msg.tokens.cache_write, 25);
        assert_eq!(msg.cost, 0.42);
        assert_eq!(msg.agent.as_deref(), Some("architect"));
        assert_eq!(msg.dedup_key.as_deref(), Some("embedded-msg-1"));
    }

    #[test]
    fn test_parse_kilo_sqlite_skips_invalid_rows_and_clamps_values() {
        let dir = TempDir::new().unwrap();
        let db_path = create_kilo_sqlite_db(&dir);
        let conn = Connection::open(&db_path).unwrap();

        insert_kilo_message(
            &conn,
            "row-user",
            "sess-user",
            r#"{
                "session_id": "sess-user",
                "role": "user",
                "modelID": "gpt-5.4",
                "tokens": {"input": 1, "output": 1, "cache": {"read": 0, "write": 0}}
            }"#,
        );
        insert_kilo_message(
            &conn,
            "row-no-tokens",
            "sess-no-tokens",
            r#"{
                "session_id": "sess-no-tokens",
                "role": "assistant",
                "modelID": "gpt-5.4"
            }"#,
        );
        insert_kilo_message(
            &conn,
            "row-no-model",
            "sess-no-model",
            r#"{
                "session_id": "sess-no-model",
                "role": "assistant",
                "tokens": {"input": 1, "output": 1, "cache": {"read": 0, "write": 0}}
            }"#,
        );
        insert_kilo_message(&conn, "row-invalid-json", "sess-invalid", "{not-json");
        insert_kilo_message(
            &conn,
            "row-valid",
            "sess-valid",
            r#"{
                "role": "assistant",
                "modelID": "gpt-5.4",
                "cost": -0.75,
                "mode": "debug",
                "tokens": {
                    "input": -100,
                    "output": -50,
                    "reasoning": -5,
                    "cache": {"read": -20, "write": -10}
                }
            }"#,
        );
        drop(conn);

        let messages = parse_kilo_sqlite_with_fallback(&db_path, 1_800_000_000_000);
        assert_eq!(messages.len(), 1);

        let msg = &messages[0];
        assert_eq!(msg.session_id, "sess-valid");
        assert_eq!(msg.model_id, "gpt-5.4");
        assert_eq!(msg.provider_id, "openai");
        assert_eq!(msg.timestamp, 1_800_000_000_000);
        assert_eq!(msg.tokens.input, 0);
        assert_eq!(msg.tokens.output, 0);
        assert_eq!(msg.tokens.reasoning, 0);
        assert_eq!(msg.tokens.cache_read, 0);
        assert_eq!(msg.tokens.cache_write, 0);
        assert_eq!(msg.cost, 0.0);
        assert_eq!(msg.agent.as_deref(), Some("debug"));
        assert_eq!(msg.dedup_key.as_deref(), Some("row-valid"));
    }

    #[test]
    fn test_parse_kilo_sqlite_returns_empty_for_missing_db() {
        let messages = parse_kilo_sqlite(std::path::Path::new("/nonexistent/kilo.db"));
        assert!(messages.is_empty());
    }

    /// Current Kilo CLI never writes `$.role` or a top-level `modelID`. Each
    /// assistant step is a `session_message` row whose `type` column is
    /// `assistant` and whose payload nests the model under `$.model`. A
    /// database that only has that table used to parse as zero usage.
    #[test]
    fn test_parse_kilo_sqlite_reads_session_message_steps() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("kilo.db");
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE session (
                id TEXT PRIMARY KEY,
                directory TEXT NOT NULL,
                title TEXT NOT NULL
            );
            CREATE TABLE session_message (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                type TEXT NOT NULL,
                data TEXT NOT NULL
            );
            INSERT INTO session (id, directory, title)
            VALUES ('ses_1', '/work/repo', 'Fix the parser');
            "#,
        )
        .unwrap();

        let step = r#"{
            "agent": "build",
            "model": {"id": "claude-sonnet-4-6", "providerID": "anthropic"},
            "cost": 0,
            "tokens": {
                "input": 800,
                "output": 120,
                "reasoning": 30,
                "cache": {"read": 400, "write": 50}
            },
            "time": {"created": 1700000000456, "completed": 1700000001456},
            "finish": "stop",
            "content": [{"type": "text", "id": "txt_1", "text": "done"}]
        }"#;
        conn.execute(
            "INSERT INTO session_message (id, session_id, type, data) VALUES (?1, ?2, ?3, ?4)",
            params!["msg_step", "ses_1", "assistant", step],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO session_message (id, session_id, type, data) VALUES (?1, ?2, ?3, ?4)",
            params![
                "msg_user",
                "ses_1",
                "user",
                r#"{"text":"hello","time":{"created":1700000000000}}"#
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO session_message (id, session_id, type, data) VALUES (?1, ?2, ?3, ?4)",
            params![
                "msg_open",
                "ses_1",
                "assistant",
                r#"{"agent":"build","model":{"id":"claude-sonnet-4-6","providerID":"anthropic"},"time":{"created":1700000002000},"content":[]}"#
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO session_message (id, session_id, type, data) VALUES (?1, ?2, ?3, ?4)",
            params!["msg_bad", "ses_1", "assistant", "{not-json"],
        )
        .unwrap();
        drop(conn);

        let messages = parse_kilo_sqlite_with_fallback(&db_path, 42);
        assert_eq!(messages.len(), 1);

        let msg = &messages[0];
        assert_eq!(msg.client, "kilo");
        assert_eq!(msg.session_id, "ses_1");
        assert_eq!(msg.model_id, "claude-sonnet-4-6");
        assert_eq!(msg.provider_id, "anthropic");
        assert_eq!(msg.timestamp, 1_700_000_000_456);
        assert_eq!(msg.tokens.input, 800);
        assert_eq!(msg.tokens.output, 120);
        assert_eq!(msg.tokens.reasoning, 30);
        assert_eq!(msg.tokens.cache_read, 400);
        assert_eq!(msg.tokens.cache_write, 50);
        assert_eq!(msg.cost, 0.0);
        assert_eq!(msg.agent.as_deref(), Some("build"));
        assert_eq!(msg.dedup_key.as_deref(), Some("msg_step"));
        assert_eq!(msg.session_title.as_deref(), Some("Fix the parser"));
        assert_eq!(msg.workspace_key.as_deref(), Some("/work/repo"));
        assert_eq!(msg.workspace_label.as_deref(), Some("repo"));
    }

    /// A current database still creates the legacy `message` table. Reading
    /// only the first table that prepares would hide whichever generation
    /// lost the race.
    #[test]
    fn test_parse_kilo_sqlite_reads_legacy_message_beside_session_message() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("kilo.db");
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE message (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                data TEXT NOT NULL
            );
            CREATE TABLE session_message (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                type TEXT NOT NULL,
                data TEXT NOT NULL
            );
            "#,
        )
        .unwrap();
        conn.execute(
            "INSERT INTO message (id, session_id, data) VALUES (?1, ?2, ?3)",
            params![
                "legacy-1",
                "ses_old",
                r#"{"role":"assistant","modelID":"gpt-5.4","providerID":"openai","cost":0.2,"tokens":{"input":10,"output":4,"reasoning":0,"cache":{"read":0,"write":0}},"time":{"created":1700000000001}}"#
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO session_message (id, session_id, type, data) VALUES (?1, ?2, ?3, ?4)",
            params![
                "msg_new",
                "ses_new",
                "assistant",
                r#"{"agent":"build","model":{"id":"claude-sonnet-4-6","providerID":"anthropic"},"cost":0,"tokens":{"input":20,"output":5,"reasoning":1,"cache":{"read":2,"write":3}},"time":{"created":1700000000002}}"#
            ],
        )
        .unwrap();
        drop(conn);

        let messages = parse_kilo_sqlite_with_fallback(&db_path, 42);
        assert_eq!(messages.len(), 2);
        let legacy = messages
            .iter()
            .find(|msg| msg.dedup_key.as_deref() == Some("legacy-1"))
            .expect("legacy message row");
        let step = messages
            .iter()
            .find(|msg| msg.dedup_key.as_deref() == Some("msg_new"))
            .expect("session_message step");
        assert_eq!(legacy.model_id, "gpt-5.4");
        assert_eq!(legacy.tokens.input, 10);
        assert_eq!(step.model_id, "claude-sonnet-4-6");
        assert_eq!(step.tokens.input, 20);
    }
}
