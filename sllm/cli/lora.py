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
import os
from argparse import Namespace, _SubParsersAction

import requests

from sllm.serve.logger import init_logger

logger = init_logger(__name__)


class LoraCommand:
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        lora_parser = parser.add_parser("lora", help="Manage LoRA adapters.")
        lora_subparsers = lora_parser.add_subparsers(dest="lora_command")

        # Load LoRA adapter
        load_parser = lora_subparsers.add_parser(
            "load", help="Load a LoRA adapter."
        )
        load_parser.add_argument(
            "--model",
            required=True,
            type=str,
            help="Model name to load LoRA adapter for.",
        )
        load_parser.add_argument(
            "--name", required=True, type=str, help="Name for the LoRA adapter."
        )
        load_parser.add_argument(
            "--path", required=True, type=str, help="Path to the LoRA adapter."
        )

        # Unload LoRA adapter
        unload_parser = lora_subparsers.add_parser(
            "unload", help="Unload a LoRA adapter."
        )
        unload_parser.add_argument(
            "--model",
            required=True,
            type=str,
            help="Model name to unload LoRA adapter from.",
        )
        unload_parser.add_argument(
            "--name",
            required=True,
            type=str,
            help="Name of the LoRA adapter to unload.",
        )

        lora_parser.set_defaults(func=LoraCommand)

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.base_url = os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8343/")

    def run(self) -> None:
        if self.args.lora_command == "load":
            self.load_lora_adapter(
                self.args.model, self.args.name, self.args.path
            )
        elif self.args.lora_command == "unload":
            self.unload_lora_adapter(self.args.model, self.args.name)
        else:
            logger.error("Unknown LoRA command")

    def load_lora_adapter(self, model: str, name: str, path: str) -> None:
        url = f"{self.base_url}v1/load_lora_adapter"
        data = {"model": model, "lora_name": name, "lora_path": path}
        headers = {"Content-Type": "application/json"}

        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            logger.info(
                f"Successfully loaded LoRA adapter '{name}' for model '{model}'"
            )
        else:
            logger.error(f"Failed to load LoRA adapter, {response.text}")

    def unload_lora_adapter(self, model: str, name: str) -> None:
        url = f"{self.base_url}v1/unload_lora_adapter"
        data = {"model": model, "lora_name": name}
        headers = {"Content-Type": "application/json"}

        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            logger.info(
                f"Successfully unloaded LoRA adapter '{name}' from model '{model}'"
            )
        else:
            logger.error(f"Failed to unload LoRA adapter, {response.text}")
