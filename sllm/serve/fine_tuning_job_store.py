# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2025                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import asyncio
from typing import Dict, Optional

import ray

from sllm.serve.logger import init_logger

logger = init_logger(__name__)


@ray.remote
class FineTuningJobStore:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}
        self.lock = asyncio.Lock()

    async def add_job(self, job_id: str, job_info: dict) -> None:
        async with self.lock:
            self.jobs[job_id] = job_info
            logger.info(f"Added job {job_id} to store")

    async def get_job(self, job_id: str) -> Optional[dict]:
        async with self.lock:
            return self.jobs.get(job_id)

    async def get_pending_jobs(self) -> Dict[str, dict]:
        async with self.lock:
            return {
                job_id: job_info
                for job_id, job_info in self.jobs.items()
                if job_info["status"] == "pending"
            }

    async def update_status(
        self, job_id: str, status: str, error: Optional[str] = None
    ) -> None:
        async with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]["status"] = status
                if error is not None:
                    self.jobs[job_id]["error"] = error
                logger.info(f"Updated job {job_id} status to {status}")

    async def delete_job(self, job_id: str) -> None:
        async with self.lock:
            if job_id in self.jobs:
                del self.jobs[job_id]
                logger.info(f"Deleted job {job_id} from store")

    async def cleanup_old_jobs(self, max_age_hours: int = 24) -> None:
        # TODO: Implement cleanup logic for old jobs
        pass
