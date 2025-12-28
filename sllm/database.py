# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #
"""
SQLite database layer for ServerlessLLM v1-beta.

Single source of truth for model configuration and node storage info.
Instance state is owned by Pylet - we only query it, never duplicate.
"""

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from sllm.logger import init_logger

logger = init_logger(__name__)

# Schema version for migrations
SCHEMA_VERSION = 2


@dataclass
class Model:
    """Model configuration and scaling state."""

    id: str  # "meta-llama/Llama-3.1-8B:vllm"
    model_name: str  # "meta-llama/Llama-3.1-8B"
    backend: str  # "vllm" or "sglang"
    status: str  # "active", "deleting"
    desired_replicas: int
    min_replicas: int
    max_replicas: int
    target_pending_requests: int
    keep_alive_seconds: int
    backend_config: Optional[Dict]
    created_at: str
    updated_at: str


@dataclass
class NodeStorage:
    """Storage info from sllm-store on a node."""

    node_name: str
    sllm_store_endpoint: Optional[str]
    cached_models: List[str]
    last_cache_update: str


class Database:
    """
    SQLite database for SLLM state persistence.

    Thread-safe via connection-per-thread pattern.
    Uses WAL mode for better concurrent read performance.
    """

    def __init__(self, db_path: str = "/var/lib/sllm/state.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-local storage for connections
        self._local = threading.local()

        # Initialize schema
        self._init_schema()

        logger.info(f"Database initialized at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                isolation_level=None,  # Autocommit mode
            )
            self._local.conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    def _init_schema(self):
        """Initialize database schema."""
        conn = self._get_connection()

        # Schema version tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            )
        """)

        # Check current version
        row = conn.execute(
            "SELECT version FROM schema_version LIMIT 1"
        ).fetchone()
        current_version = row[0] if row else 0

        if current_version < SCHEMA_VERSION:
            self._migrate(conn, current_version)

    def _migrate(self, conn: sqlite3.Connection, from_version: int):
        """Run migrations from from_version to SCHEMA_VERSION."""
        if from_version < 1:
            self._migrate_v1(conn)
        if from_version < 2:
            self._migrate_v2(conn)

        # Update schema version
        conn.execute("DELETE FROM schema_version")
        conn.execute(
            "INSERT INTO schema_version (version) VALUES (?)",
            (SCHEMA_VERSION,),
        )
        logger.info(f"Database migrated to schema version {SCHEMA_VERSION}")

    def _migrate_v1(self, conn: sqlite3.Connection):
        """Create initial v1 schema."""
        # Models table - model configuration and scaling state
        conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                backend TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                desired_replicas INTEGER DEFAULT 0,
                min_replicas INTEGER DEFAULT 0,
                max_replicas INTEGER DEFAULT 1,
                target_pending_requests INTEGER DEFAULT 5,
                keep_alive_seconds INTEGER DEFAULT 0,
                backend_config TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Index for status queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_models_status
            ON models(status)
        """)

        # Node storage table - cache info from sllm-store
        conn.execute("""
            CREATE TABLE IF NOT EXISTS node_storage (
                node_name TEXT PRIMARY KEY,
                sllm_store_endpoint TEXT,
                cached_models TEXT,
                last_cache_update TEXT NOT NULL
            )
        """)

        logger.info("Created v1 schema tables")

    def _migrate_v2(self, conn: sqlite3.Connection):
        """Add model_endpoints table for router."""
        # Model endpoints table - tracks healthy endpoints for each model
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_endpoints (
                model_id TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'healthy',
                added_at TEXT NOT NULL,
                PRIMARY KEY (model_id, endpoint)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_endpoints_model_id
            ON model_endpoints(model_id)
        """)

        logger.info("Created v2 schema: model_endpoints table")

    # -------------------------------------------------------------------------
    # Model CRUD Operations
    # -------------------------------------------------------------------------

    def create_model(
        self,
        model_id: str,
        model_name: str,
        backend: str,
        min_replicas: int = 0,
        max_replicas: int = 1,
        target_pending_requests: int = 5,
        keep_alive_seconds: int = 0,
        backend_config: Optional[Dict] = None,
    ) -> Model:
        """Create a new model entry."""
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()

        backend_config_json = (
            json.dumps(backend_config) if backend_config else None
        )

        try:
            conn.execute(
                """
                INSERT INTO models (
                    id, model_name, backend, status, desired_replicas,
                    min_replicas, max_replicas, target_pending_requests,
                    keep_alive_seconds, backend_config, created_at, updated_at
                ) VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    model_name,
                    backend,
                    min_replicas,  # desired starts at min
                    min_replicas,
                    max_replicas,
                    target_pending_requests,
                    keep_alive_seconds,
                    backend_config_json,
                    now,
                    now,
                ),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"Model {model_id} already exists")

        logger.info(f"Created model {model_id}")
        return self.get_model(model_id)

    def get_model(self, model_id: str) -> Optional[Model]:
        """Get a model by ID."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM models WHERE id = ?", (model_id,)
        ).fetchone()

        if not row:
            return None

        return self._row_to_model(row)

    def get_all_models(self) -> List[Model]:
        """Get all models."""
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM models").fetchall()
        return [self._row_to_model(row) for row in rows]

    def get_active_models(self) -> List[Model]:
        """Get all active (non-deleting) models."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM models WHERE status = 'active'"
        ).fetchall()
        return [self._row_to_model(row) for row in rows]

    def update_desired_replicas(self, model_id: str, desired: int) -> bool:
        """Update desired_replicas for a model. Returns True if updated."""
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()

        cursor = conn.execute(
            """
            UPDATE models
            SET desired_replicas = ?, updated_at = ?
            WHERE id = ? AND status = 'active'
            """,
            (desired, now, model_id),
        )

        return cursor.rowcount > 0

    def update_model_status(self, model_id: str, status: str) -> bool:
        """Update model status. Returns True if updated."""
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()

        cursor = conn.execute(
            """
            UPDATE models
            SET status = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, now, model_id),
        )

        if cursor.rowcount > 0:
            logger.info(f"Model {model_id} status changed to {status}")
            return True
        return False

    def delete_model(self, model_id: str) -> bool:
        """Delete a model. Returns True if deleted."""
        conn = self._get_connection()
        cursor = conn.execute("DELETE FROM models WHERE id = ?", (model_id,))

        if cursor.rowcount > 0:
            logger.info(f"Deleted model {model_id}")
            return True
        return False

    def _row_to_model(self, row: sqlite3.Row) -> Model:
        """Convert a database row to a Model object."""
        backend_config = None
        if row["backend_config"]:
            backend_config = json.loads(row["backend_config"])

        return Model(
            id=row["id"],
            model_name=row["model_name"],
            backend=row["backend"],
            status=row["status"],
            desired_replicas=row["desired_replicas"],
            min_replicas=row["min_replicas"],
            max_replicas=row["max_replicas"],
            target_pending_requests=row["target_pending_requests"],
            keep_alive_seconds=row["keep_alive_seconds"],
            backend_config=backend_config,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # -------------------------------------------------------------------------
    # Node Storage Operations
    # -------------------------------------------------------------------------

    def upsert_node_storage(
        self,
        node_name: str,
        sllm_store_endpoint: Optional[str],
        cached_models: List[str],
    ):
        """Insert or update node storage info."""
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()
        cached_models_json = json.dumps(cached_models)

        conn.execute(
            """
            INSERT INTO node_storage (
                node_name, sllm_store_endpoint, cached_models, last_cache_update
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(node_name) DO UPDATE SET
                sllm_store_endpoint = excluded.sllm_store_endpoint,
                cached_models = excluded.cached_models,
                last_cache_update = excluded.last_cache_update
            """,
            (node_name, sllm_store_endpoint, cached_models_json, now),
        )

        logger.debug(
            f"Updated storage for node {node_name}: "
            f"{len(cached_models)} cached models"
        )

    def get_node_storage(self, node_name: str) -> Optional[NodeStorage]:
        """Get storage info for a node."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM node_storage WHERE node_name = ?", (node_name,)
        ).fetchone()

        if not row:
            return None

        return self._row_to_node_storage(row)

    def get_all_node_storage(self) -> List[NodeStorage]:
        """Get storage info for all nodes."""
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM node_storage").fetchall()
        return [self._row_to_node_storage(row) for row in rows]

    def get_nodes_with_model(self, model_name: str) -> List[str]:
        """Get list of nodes that have a model cached."""
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM node_storage").fetchall()

        nodes = []
        for row in rows:
            cached = (
                json.loads(row["cached_models"]) if row["cached_models"] else []
            )
            if model_name in cached:
                nodes.append(row["node_name"])

        return nodes

    def delete_node_storage(self, node_name: str) -> bool:
        """Delete storage info for a node. Returns True if deleted."""
        conn = self._get_connection()
        cursor = conn.execute(
            "DELETE FROM node_storage WHERE node_name = ?", (node_name,)
        )
        return cursor.rowcount > 0

    def _row_to_node_storage(self, row: sqlite3.Row) -> NodeStorage:
        """Convert a database row to a NodeStorage object."""
        cached_models = []
        if row["cached_models"]:
            cached_models = json.loads(row["cached_models"])

        return NodeStorage(
            node_name=row["node_name"],
            sllm_store_endpoint=row["sllm_store_endpoint"],
            cached_models=cached_models,
            last_cache_update=row["last_cache_update"],
        )

    # -------------------------------------------------------------------------
    # Model Endpoints Operations (for Router)
    # -------------------------------------------------------------------------

    def get_model_endpoints(self, model_id: str) -> List[str]:
        """Get healthy endpoints for a model. Called by Router."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT endpoint FROM model_endpoints "
            "WHERE model_id = ? AND status = 'healthy'",
            (model_id,),
        ).fetchall()
        return [row[0] for row in rows]

    def add_model_endpoint(self, model_id: str, endpoint: str):
        """Add endpoint. Called by Reconciler."""
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO model_endpoints "
            "(model_id, endpoint, status, added_at) VALUES (?, ?, 'healthy', ?)",
            (model_id, endpoint, now),
        )
        logger.debug(f"Added endpoint {endpoint} for {model_id}")

    def remove_model_endpoint(self, model_id: str, endpoint: str):
        """Remove endpoint. Called by Reconciler."""
        conn = self._get_connection()
        cursor = conn.execute(
            "DELETE FROM model_endpoints WHERE model_id = ? AND endpoint = ?",
            (model_id, endpoint),
        )
        if cursor.rowcount > 0:
            logger.debug(f"Removed endpoint {endpoint} for {model_id}")

    def mark_endpoint_unhealthy(self, model_id: str, endpoint: str):
        """Mark endpoint unhealthy. Called by Reconciler."""
        conn = self._get_connection()
        conn.execute(
            "UPDATE model_endpoints SET status = 'unhealthy' "
            "WHERE model_id = ? AND endpoint = ?",
            (model_id, endpoint),
        )
        logger.debug(f"Marked endpoint {endpoint} unhealthy for {model_id}")

    def remove_model_endpoints(self, model_id: str):
        """Remove all endpoints for a model. Called during model deletion."""
        conn = self._get_connection()
        cursor = conn.execute(
            "DELETE FROM model_endpoints WHERE model_id = ?",
            (model_id,),
        )
        if cursor.rowcount > 0:
            logger.debug(f"Removed {cursor.rowcount} endpoints for {model_id}")

    def get_all_endpoints_for_model(self, model_id: str) -> List[dict]:
        """Get all endpoints (including unhealthy) for a model."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT endpoint, status, added_at FROM model_endpoints "
            "WHERE model_id = ?",
            (model_id,),
        ).fetchall()
        return [
            {"endpoint": row[0], "status": row[1], "added_at": row[2]}
            for row in rows
        ]

    def get_all_healthy_endpoints(self) -> Dict[str, List[str]]:
        """Get all healthy endpoints grouped by model ID."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT model_id, endpoint FROM model_endpoints "
            "WHERE status = 'healthy'"
        ).fetchall()

        result: Dict[str, List[str]] = {}
        for row in rows:
            model_id, endpoint = row[0], row[1]
            if model_id not in result:
                result[model_id] = []
            result[model_id].append(endpoint)
        return result

    def delete_model_endpoints(self, model_id: str):
        """Alias for remove_model_endpoints (for test compatibility)."""
        return self.remove_model_endpoints(model_id)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def close(self):
        """Close the database connection for the current thread."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def reset(self):
        """Reset database - delete all data. Use with caution!"""
        conn = self._get_connection()
        conn.execute("DELETE FROM models")
        conn.execute("DELETE FROM node_storage")
        conn.execute("DELETE FROM model_endpoints")
        logger.warning("Database reset - all data deleted")


# Global database instance (initialized on first use)
_db: Optional[Database] = None
_db_lock = threading.Lock()


def get_database(db_path: Optional[str] = None) -> Database:
    """Get the global database instance."""
    global _db
    with _db_lock:
        if _db is None:
            path = db_path or "/var/lib/sllm/state.db"
            _db = Database(path)
        return _db


def init_database(db_path: str) -> Database:
    """Initialize the global database instance with a specific path."""
    global _db
    with _db_lock:
        if _db is not None:
            _db.close()
        _db = Database(db_path)
        return _db
