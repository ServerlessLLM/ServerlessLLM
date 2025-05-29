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

from sllm.cli._cli_utils import read_config, validate_config
from sllm.serve.logger import init_logger

logger = init_logger(__name__)


class UpdateCommand:
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        update_parser = parser.add_parser(
            "update", help="Update a model using a config file or model name."
        )
        update_parser.add_argument(
            "--model",
            type=str,
            help="Model name to update with new configuration.",
        )
        update_parser.add_argument(
            "--config", type=str, help="Path to the JSON config file."
        )
        update_parser.set_defaults(func=UpdateCommand)

    def __init__(self, args: Namespace) -> None:
        self.model = args.model
        self.config_path = args.config
        self.url = (
            os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8343") + "/update"
        )
        self.default_config_path = os.path.join(
            os.path.dirname(__file__), "default_config.json"
        )

    def run(self) -> None:
        if self.config_path:
            config_data = read_config(self.config_path)
            validate_config(config_data)
            self.update_model(config_data)
        elif self.model:
            config_data = read_config(self.default_config_path)
            config_data["model"] = self.model
            config_data["backend_config"]["pretrained_model_name_or_path"] = (
                self.model
            )
            logger.info(
                f"Updating model {self.model} with default configuration."
            )
            self.update_model(config_data)
        else:
            logger.error("You must specify either --model or --config.")
            exit(1)

    def update_model(self, config_data: dict) -> None:
        headers = {"Content-Type": "application/json"}

        # Send POST request to the /update endpoint
        response = requests.post(self.url, headers=headers, json=config_data)

        if response.status_code == 200:
            logger.info("Model updated successfully.")
        else:
            logger.error(
                f"Failed to update model. Status code: {response.status_code}. Response: {response.text}"
            )
