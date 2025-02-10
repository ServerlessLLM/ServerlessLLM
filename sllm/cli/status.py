# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
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
import os
from argparse import Namespace, _SubParsersAction

import requests

from sllm.cli._cli_utils import read_config
from sllm.serve.logger import init_logger

logger = init_logger(__name__)


class StatusCommand:
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        status_parser = parser.add_parser(
            "status",
            help="Query the information of registered models.",
        )
        status_parser.set_defaults(func=StatusCommand)

    def __init__(self, args: Namespace) -> None:
        self.endpoint = "v1/models"  # TODO: as a argument
        self.url = (
            os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8343/")
            + self.endpoint
        )

    def run(self) -> None:
        status = self.query_status()
        if status:
            print(f"Model status: {status}")
        else:
            logger.error("Failed to fetch model status.")

    def query_status(self) -> dict:
        headers = {"Content-Type": "application/json"}
        try:
            # Send GET request to the status endpoint
            response = requests.get(self.url, headers=headers)

            if response.status_code == 200:
                logger.info("Status query successful.")
                try:
                    return response.json()
                except ValueError:
                    logger.error("Invalid JSON response received.")
                    return None
            else:
                logger.error(
                    f"Failed to query status. Status code: {response.status_code}"
                )
                logger.error(f"Response: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed:{str(e)}")
            return None
