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

from sllm.serve.logger import init_logger

logger = init_logger(__name__)


class DeleteCommand:
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        delete_parser = parser.add_parser(
            "delete", help="Delete deployed models by name."
        )
        delete_parser.add_argument(
            "models", nargs="+", type=str, help="Model names to delete."
        )
        delete_parser.add_argument(
            "--lora-adapters",
            nargs="+",
            type=str,
            help="LoRA adapters to delete.",
        )
        delete_parser.set_defaults(func=DeleteCommand)

    def __init__(self, args: Namespace) -> None:
        self.models = args.models
        self.lora_adapters = getattr(args, "lora_adapters", None)
        self.url = (
            os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8343/") + "delete/"
        )

    def run(self) -> None:
        headers = {"Content-Type": "application/json"}
        if self.lora_adapters is not None and len(self.models) > 1:
            logger.error(
                "You can only delete one model when using --lora-adapters."
            )
            exit(1)

        for model in self.models:
            data = {"model": model}
            if self.lora_adapters is not None:
                data["lora_adapters"] = self.lora_adapters
            response = requests.post(self.url, headers=headers, json=data)

            if response.status_code == 200:
                logger.info(
                    f"Successfully sent the request to delete model {model}"
                )
            else:
                logger.error(
                    f"Failed to delete model: {model}. Status code: {response.status_code}. Response: {response.text}"
                )
