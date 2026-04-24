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

Single source of truth for deployment configuration and node storage info.
Instance state is owned by Pylet - we only query it, never duplicate.

Terminology:
- Deployment: A (model_name, backend) pair - the basic scheduling unit
- deployment_id: Unique identifier for a deployment (format: "{model_name}:{backend}")
- model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.1-8B")
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
SCHEMA_VERSION = 5

@dataclass
class Deployment:
    """Deployment configuration and scaling state.
    
    A deployment represents a (model_name, backend) pair - the basic
    scheduling and control unit in ServerlessLLM.
    """
    id: str  # deployment_id: "meta-llama/Llama-3.1-8B:vllm"
    model_name: str  # HuggingFace model: "meta-llama/Llama-3.1-8B"
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

    @staticmethod
    def make_id(model_name: str, backend: str) -> str:
        """Generate deployment_id from model_name and backend.

        This is the single source of truth for deployment ID format.
        """
        return f"{model_name}:{backend}"

@dataclass
class NodeStorage:
    """Storage info from sllm-store on a node."""
    node_name: str
    sllm_store_endpoint: Optional[str]
    cached_models: List[str]
    last_cache_update: str

@dataclass
class FileObject:
    """A file uploaded by the user."""
    id: str  # e.g., "file-..."
    filename: str
    bytes: int
    purpose: str
    created_at: str

@dataclass
class BatchJob:
    """A batch job containing multiple tasks."""
    id: str
    status: str
    metadata: Optional[Dict]
    created_at: str
    updated_at: str
    input_file_id: Optional[str] = None


@dataclass
class BatchTask:
    """A single task within a batch job."""

    id: str  # UUID
    batch_id: str
    custom_id: str  # User-provided ID
    method: str
    url: str
    body: Dict
    status: str  # "pending", "completed", "failed"
    output: Optional[Dict]
    created_at: str
    updated_at: str
    # Metrics
    started_at: Optional[str] = None      # When execution started (sent to vLLM)
    completed_at: Optional[str] = None    # When execution finished


class Database:
    """
    SQLite database for SLLM state persistence.
    
    Thread-safe via connection-per-thread pattern.
    Uses WAL mode for better concurrent read performance.
    """

    # -------------------------------------------------------------------------
    # Batch Job Operations
    # -------------------------------------------------------------------------

    def update_batch_job_status(self, batch_id: str, status: str):
        """Update batch job status."""
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE batch_jobs SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, batch_id)
        )
        logger.debug(f"Updated batch {batch_id} status to {status}")

    def update_batch_task_result(
        self, task_id: str, status: str, output: Optional[Dict] = None
    ):
        """Update task status and output."""
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()
        output_json = json.dumps(output) if output else None
        
        conn.execute(
            """
            UPDATE batch_tasks 
            SET status = ?, output = ?, updated_at = ? 
            WHERE id = ?
            """,
            (status, output_json, now, task_id)
        )

    def get_pending_batch_ids(self) -> List[str]:
        """Get IDs of all batches that are pending or in_progress."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT id FROM batch_jobs WHERE status IN ('pending', 'in_progress')"
        ).fetchall()
        return [row[0] for row in rows]

    def create_batch_job(
        self, batch_id: str, metadata: Optional[Dict] = None, input_file_id: Optional[str] = None
    ) -> "BatchJob":
        """Create a new batch job."""
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(metadata) if metadata else None

        conn.execute(
            "INSERT INTO batch_jobs (id, status, metadata, input_file_id, created_at, updated_at) "
            "VALUES (?, 'pending', ?, ?, ?, ?)",
            (batch_id, metadata_json, input_file_id, now, now),
        )
        return BatchJob(
            id=batch_id,
            status="pending",
            metadata=metadata,
            input_file_id=input_file_id,
            created_at=now,
            updated_at=now,
        )

    def get_batch_job(self, batch_id: str) -> Optional["BatchJob"]:
        """Get a batch job by ID."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM batch_jobs WHERE id = ?", (batch_id,)
        ).fetchone()

        if not row:
            return None
        return self._row_to_batch_job(row)

    def _row_to_batch_job(self, row: sqlite3.Row) -> "BatchJob":
        metadata = None
        if "metadata" in row.keys() and row["metadata"]:
            metadata = json.loads(row["metadata"])
            
        return BatchJob(
            id=row["id"],
            status=row["status"],
            metadata=metadata,
            input_file_id=row["input_file_id"] if "input_file_id" in row.keys() else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # -------------------------------------------------------------------------
    # File Management Operations
    # -------------------------------------------------------------------------

    def create_file(self, file_id: str, filename: str, bytes_size: int, purpose: str) -> FileObject:
        """Create a new file record."""
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO files (id, filename, bytes, purpose, created_at) VALUES (?, ?, ?, ?, ?)",
            (file_id, filename, bytes_size, purpose, now)
        )
        logger.info(f"Created file {file_id}")
        return FileObject(
            id=file_id,
            filename=filename,
            bytes=bytes_size,
            purpose=purpose,
            created_at=now
        )

    def get_file(self, file_id: str) -> Optional[FileObject]:
        """Get file by ID."""
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
        if not row:
            return None
        return FileObject(
            id=row["id"],
            filename=row["filename"],
            bytes=row["bytes"],
            purpose=row["purpose"],
            created_at=row["created_at"]
        )

    def get_all_files(self) -> List[FileObject]:
        """Get all files."""
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM files ORDER BY created_at DESC").fetchall()
        return [
            FileObject(
                id=row["id"],
                filename=row["filename"],
                bytes=row["bytes"],
                purpose=row["purpose"],
                created_at=row["created_at"]
            )
            for row in rows
        ]

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
        # v3 is a breaking change - fail if old schema exists
        if from_version > 0 and from_version < 3:
            raise RuntimeError(
                f"Database schema v{from_version} is incompatible with v3. "
                f"Please delete {self.db_path} and restart. "
                "This is expected during v1-beta development."
            )

        if from_version < 3:
            self._migrate_v3(conn)
        if from_version < 4:
            self._migrate_v4(conn)
        if from_version < 5:
            self._migrate_v5(conn)

        # Update schema version
        conn.execute("DELETE FROM schema_version")
        conn.execute(
            "INSERT INTO schema_version (version) VALUES (?)",
            (SCHEMA_VERSION,),
        )
        logger.info(f"Database migrated to schema version {SCHEMA_VERSION}")

    def _migrate_v3(self, conn: sqlite3.Connection):
        """Create v3 schema with deployment terminology."""
        # Deployments table - deployment configuration and scaling state
        conn.execute("""
            CREATE TABLE IF NOT EXISTS deployments (
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
            CREATE INDEX IF NOT EXISTS idx_deployments_status
            ON deployments(status)
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

        # Deployment endpoints table - tracks healthy endpoints per deployment
        conn.execute("""
            CREATE TABLE IF NOT EXISTS deployment_endpoints (
                deployment_id TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'healthy',
                added_at TEXT NOT NULL,
                PRIMARY KEY (deployment_id, endpoint)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_deployment_endpoints_deployment_id
            ON deployment_endpoints(deployment_id)
        """)

        logger.info("Created v3 schema with deployment terminology")

    def _migrate_v4(self, conn: sqlite3.Connection):
        """Create v4 schema with batch job support (Squashed v4-v6)."""
        # Batch Jobs table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS batch_jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Batch Tasks table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS batch_tasks (
                id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL,
                custom_id TEXT NOT NULL,
                method TEXT NOT NULL,
                url TEXT NOT NULL,
                body TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                output TEXT,
                started_at TEXT,
                completed_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(batch_id) REFERENCES batch_jobs(id)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_batch_tasks_batch_id
            ON batch_tasks(batch_id)
        """)

        logger.info("Created v4 schema (Batch Support)")

    def _migrate_v5(self, conn: sqlite3.Connection):
        """Create v5 schema with file management support."""
        # Files table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                bytes INTEGER NOT NULL,
                purpose TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        # Alter batch_jobs to add input_file_id (if not exists)
        try:
            conn.execute("ALTER TABLE batch_jobs ADD COLUMN input_file_id TEXT")
        except sqlite3.OperationalError:
            pass # Column already exists, which is fine

        logger.info("Created v5 schema (File Support)")

    # -------------------------------------------------------------------------
    # Deployment CRUD Operations
    # -------------------------------------------------------------------------

    def create_deployment(
        self,
        model_name: str,
        backend: str,
        min_replicas: int = 0,
        max_replicas: int = 1,
        target_pending_requests: int = 5,
        keep_alive_seconds: int = 0,
        backend_config: Optional[Dict] = None,
    ) -> Deployment:
        """Create a new deployment entry."""
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()
        deployment_id = Deployment.make_id(model_name, backend)

        backend_config_json = (
            json.dumps(backend_config) if backend_config else None
        )

        try:
            conn.execute(
                """
                INSERT INTO deployments (
                    id, model_name, backend, status, desired_replicas,
                    min_replicas, max_replicas, target_pending_requests,
                    keep_alive_seconds, backend_config, created_at, updated_at
                ) VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    deployment_id,
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
            raise ValueError(f"Deployment {deployment_id} already exists")

        logger.info(f"Created deployment {deployment_id}")
        return self.get_deployment(model_name, backend)

    def get_deployment(
        self, model_name: str, backend: str
    ) -> Optional[Deployment]:
        """Get a deployment by model_name and backend."""
        deployment_id = Deployment.make_id(model_name, backend)
        return self.get_deployment_by_id(deployment_id)

    def get_deployment_by_id(self, deployment_id: str) -> Optional[Deployment]:
        """Get a deployment by its ID."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM deployments WHERE id = ?", (deployment_id,)
        ).fetchone()

        if not row:
            return None

        return self._row_to_deployment(row)

    def get_all_deployments(self) -> List[Deployment]:
        """Get all deployments."""
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM deployments").fetchall()
        return [self._row_to_deployment(row) for row in rows]

    def get_active_deployments(self) -> List[Deployment]:
        """Get all active (non-deleting) deployments."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM deployments WHERE status = 'active'"
        ).fetchall()
        return [self._row_to_deployment(row) for row in rows]

    def update_max_replicas(self, deployment_id: str, max_replicas: int) -> bool:
        """Update max_replicas for a deployment. Returns True if updated."""
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()

        cursor = conn.execute(
            """
            UPDATE deployments
            SET max_replicas = ?, updated_at = ?
            WHERE id = ? AND status = 'active'
            """,
            (max_replicas, now, deployment_id),
        )

        return cursor.rowcount > 0

    def update_desired_replicas(self, deployment_id: str, desired: int) -> bool:
        """Update desired_replicas for a deployment. Returns True if updated."""
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()

        cursor = conn.execute(
            """
            UPDATE deployments
            SET desired_replicas = ?, updated_at = ?
            WHERE id = ? AND status = 'active'
            """,
            (desired, now, deployment_id),
        )

        return cursor.rowcount > 0

    def update_deployment_status(self, deployment_id: str, status: str) -> bool:
        """Update deployment status. Returns True if updated."""
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()

        cursor = conn.execute(
            """
            UPDATE deployments
            SET status = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, now, deployment_id),
        )

        if cursor.rowcount > 0:
            logger.info(
                f"Deployment {deployment_id} status changed to {status}"
            )
            return True
        return False

    def delete_deployment(self, deployment_id: str) -> bool:
        """Delete a deployment. Returns True if deleted."""
        conn = self._get_connection()
        cursor = conn.execute(
            "DELETE FROM deployments WHERE id = ?", (deployment_id,)
        )

        if cursor.rowcount > 0:
            logger.info(f"Deleted deployment {deployment_id}")
            return True
        return False

    def _row_to_deployment(self, row: sqlite3.Row) -> Deployment:
        """Convert a database row to a Deployment object."""
        backend_config = None
        if row["backend_config"]:
            backend_config = json.loads(row["backend_config"])

        return Deployment(
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
    # Batch Job Operations
    # -------------------------------------------------------------------------

    # Job methods defined above...

    def create_batch_task(
        self,
        task_id: str,
        batch_id: str,
        custom_id: str,
        method: str,
        url: str,
        body: Dict,
    ) -> "BatchTask":
        """Create a new batch task."""
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()
        body_json = json.dumps(body)

        conn.execute(
            """
            INSERT INTO batch_tasks (
                id, batch_id, custom_id, method, url, body, status,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
            """,
            (
                task_id,
                batch_id,
                custom_id,
                method,
                url,
                body_json,
                now,
                now,
            ),
        )
        return BatchTask(
            id=task_id,
            batch_id=batch_id,
            custom_id=custom_id,
            method=method,
            url=url,
            body=body,
            status="pending",
            output=None,
            created_at=now,
            updated_at=now,
        )

    def create_batch_tasks_bulk(
        self,
        tasks: List[Dict],
        batch_id: str,
    ) -> int:
        """Bulk-insert batch tasks in a single transaction.

        Args:
            tasks: List of dicts with keys: task_id, custom_id, method, url, body
            batch_id: Parent batch job ID

        Returns:
            Number of tasks inserted.
        """
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()

        rows = [
            (
                t["task_id"],
                batch_id,
                t["custom_id"],
                t["method"],
                t["url"],
                json.dumps(t["body"]),
                now,
                now,
            )
            for t in tasks
        ]

        conn.execute("BEGIN")
        try:
            conn.executemany(
                """
                INSERT INTO batch_tasks (
                    id, batch_id, custom_id, method, url, body, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                rows,
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        return len(rows)

    def upsert_batch_task(
        self,
        task_id: str,
        batch_id: str,
        custom_id: str,
        method: str,
        url: str,
        body: Dict,
        status: str = "pending",
        output: Optional[Dict] = None,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
    ):
        """Insert or update a batch task with metrics support."""
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()
        body_json = json.dumps(body)
        output_json = json.dumps(output) if output else None

        conn.execute(
            """
            INSERT INTO batch_tasks (
                id, batch_id, custom_id, method, url, body, status,
                output, started_at, completed_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                output = excluded.output,
                started_at = COALESCE(excluded.started_at, batch_tasks.started_at),
                completed_at = COALESCE(excluded.completed_at, batch_tasks.completed_at),
                updated_at = excluded.updated_at
            """,
            (
                task_id,
                batch_id,
                custom_id,
                method,
                url,
                body_json,
                status,
                output_json,
                started_at,
                completed_at,
                now,
                now,
            ),
        )

    def get_batch_tasks(self, batch_id: str) -> List["BatchTask"]:
        """Get all tasks for a batch job."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM batch_tasks WHERE batch_id = ?", (batch_id,)
        ).fetchall()
        return [self._row_to_batch_task(row) for row in rows]

    def _row_to_batch_task(self, row: sqlite3.Row) -> "BatchTask":
        return BatchTask(
            id=row["id"],
            batch_id=row["batch_id"],
            custom_id=row["custom_id"],
            method=row["method"],
            url=row["url"],
            body=json.loads(row["body"]),
            status=row["status"],
            output=json.loads(row["output"]) if row["output"] else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"] if "started_at" in row.keys() else None,
            completed_at=row["completed_at"] if "completed_at" in row.keys() else None,
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
    # Deployment Endpoints Operations (for Router)
    # -------------------------------------------------------------------------

    def get_deployment_endpoints(self, deployment_id: str) -> List[str]:
        """Get healthy endpoints for a deployment. Called by Router."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT endpoint FROM deployment_endpoints "
            "WHERE deployment_id = ? AND status = 'healthy' "
            "ORDER BY endpoint",
            (deployment_id,),
        ).fetchall()
        return [row[0] for row in rows]

    def add_deployment_endpoint(self, deployment_id: str, endpoint: str):
        """Add endpoint. Called by Reconciler."""
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO deployment_endpoints "
            "(deployment_id, endpoint, status, added_at) "
            "VALUES (?, ?, 'healthy', ?)",
            (deployment_id, endpoint, now),
        )
        logger.debug(f"Added endpoint {endpoint} for {deployment_id}")

    def remove_deployment_endpoint(self, deployment_id: str, endpoint: str):
        """Remove endpoint. Called by Reconciler."""
        conn = self._get_connection()
        cursor = conn.execute(
            "DELETE FROM deployment_endpoints "
            "WHERE deployment_id = ? AND endpoint = ?",
            (deployment_id, endpoint),
        )
        if cursor.rowcount > 0:
            logger.debug(f"Removed endpoint {endpoint} for {deployment_id}")

    def mark_endpoint_unhealthy(self, deployment_id: str, endpoint: str):
        """Mark endpoint unhealthy. Called by Reconciler."""
        conn = self._get_connection()
        conn.execute(
            "UPDATE deployment_endpoints SET status = 'unhealthy' "
            "WHERE deployment_id = ? AND endpoint = ?",
            (deployment_id, endpoint),
        )
        logger.debug(
            f"Marked endpoint {endpoint} unhealthy for {deployment_id}"
        )

    def remove_deployment_endpoints(self, deployment_id: str):
        """Remove all endpoints for a deployment. Called during deletion."""
        conn = self._get_connection()
        cursor = conn.execute(
            "DELETE FROM deployment_endpoints WHERE deployment_id = ?",
            (deployment_id,),
        )
        if cursor.rowcount > 0:
            logger.debug(
                f"Removed {cursor.rowcount} endpoints for {deployment_id}"
            )

    def get_all_endpoints_for_deployment(
        self, deployment_id: str
    ) -> List[dict]:
        """Get all endpoints (including unhealthy) for a deployment."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT endpoint, status, added_at FROM deployment_endpoints "
            "WHERE deployment_id = ?",
            (deployment_id,),
        ).fetchall()
        return [
            {"endpoint": row[0], "status": row[1], "added_at": row[2]}
            for row in rows
        ]

    def get_all_healthy_endpoints(self) -> Dict[str, List[str]]:
        """Get all healthy endpoints grouped by deployment ID."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT deployment_id, endpoint FROM deployment_endpoints "
            "WHERE status = 'healthy'"
        ).fetchall()

        result: Dict[str, List[str]] = {}
        for row in rows:
            deployment_id, endpoint = row[0], row[1]
            if deployment_id not in result:
                result[deployment_id] = []
            result[deployment_id].append(endpoint)
        return result

    def delete_deployment_endpoints(self, deployment_id: str):
        """Alias for remove_deployment_endpoints (for test compatibility)."""
        return self.remove_deployment_endpoints(deployment_id)

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
        conn.execute("DELETE FROM deployments")
        conn.execute("DELETE FROM node_storage")
        conn.execute("DELETE FROM deployment_endpoints")
        conn.execute("DELETE FROM batch_jobs")
        conn.execute("DELETE FROM batch_tasks")
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
