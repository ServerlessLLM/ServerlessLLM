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


class DeployCommand:
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        deploy_parser = parser.add_parser(
            "deploy", help="Deploy a model using a config file or model name."
        )
        deploy_parser.add_argument(
            "--model",
            type=str,
            help="Model name to deploy with default configuration.",
        )
        deploy_parser.add_argument(
            "--config", type=str, help="Path to the JSON config file."
        )
        deploy_parser.add_argument(
            "--backend",
            type=str,
            help="Overwrite the backend in the default configuration.",
        )
        deploy_parser.add_argument(
            "--num-gpus",
            type=int,
            help="Overwrite the number of GPUs in the default configuration.",
        )
        deploy_parser.add_argument(
            "--target",
            type=int,
            help="Overwrite the target concurrency in the default configuration.",
        )
        deploy_parser.add_argument(
            "--min-instances",
            type=int,
            help="Overwrite the minimum instances in the default configuration.",
        )
        deploy_parser.add_argument(
            "--max-instances",
            type=int,
            help="Overwrite the maximum instances in the default configuration.",
        )
        deploy_parser.add_argument(
            "--enable_lora",
            action="store_true",
            help="Enable LoRA adapter support.",
        )
        deploy_parser.add_argument(
            "--lora_adapters",
            type=str,
            nargs="+",
            help="Specify LoRA adapters in the format <name>=<path>.",
        )
        deploy_parser.set_defaults(func=DeployCommand)

    def __init__(self, args: Namespace) -> None:
        self.model = args.model
        self.config_path = args.config
        self.backend = args.backend
        self.num_gpus = args.num_gpus
        self.target = args.target
        self.min_instances = args.min_instances
        self.max_instances = args.max_instances
        self.enable_lora = args.enable_lora
        self.lora_adapters = self.parse_lora_adapters(args.lora_adapters)
        self.url = (
            os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8343/") + "register"
        )
        self.default_config_path = os.path.join(
            os.path.dirname(__file__), "default_config.json"
        )

    def parse_lora_adapters(self, lora_adapters):
        parsed_modules = {}
        if lora_adapters:
            for module in lora_adapters:
                if "=" not in module:
                    raise ValueError(
                        f"Invalid LoRA module format: {module}. Expected <name>=<path>."
                    )
                name, path = module.split("=", 1)
                parsed_modules[name] = path
        return parsed_modules

    def validate_config(self, config_data: dict) -> None:
        """Validate the provided configuration data to ensure correctness."""
        try:
            num_gpus = config_data["num_gpus"]
            target = config_data["auto_scaling_config"]["target"]
            min_instances = config_data["auto_scaling_config"]["min_instances"]
            max_instances = config_data["auto_scaling_config"]["max_instances"]
        except KeyError as e:
            raise ValueError(f"Missing key in config_data: {e}")

        if num_gpus < 1:
            raise ValueError("Number of GPUs cannot be less than 1.")
        if target < 1:
            raise ValueError("Target concurrency cannot be less than 1.")
        if min_instances < 0:
            raise ValueError("Minimum instances cannot be negative.")
        if max_instances < 0:
            raise ValueError("Maximum instances cannot be negative.")
        if min_instances > max_instances:
            raise ValueError(
                "Minimum instances cannot be greater than maximum instances."
            )

    def update_config(
        self, default_config: dict, provided_config: dict
    ) -> dict:
        """Update the default configuration with values from the provided configuration."""
        for key, value in provided_config.items():
            if isinstance(value, dict) and key in default_config:
                default_config[key] = self.update_config(
                    default_config[key], value
                )
            else:
                default_config[key] = value
        return default_config

    def run(self) -> None:
        default_config = read_config(self.default_config_path)

        if self.config_path:
            provided_config = read_config(self.config_path)
            config_data = self.update_config(default_config, provided_config)
            # If pretrained_model_name_or_path is not provided, use the model name
            if (
                config_data["backend_config"]["pretrained_model_name_or_path"]
                == ""
            ):
                config_data["backend_config"][
                    "pretrained_model_name_or_path"
                ] = config_data["model"]
        elif self.model:
            config_data = default_config
            config_data["model"] = self.model
            config_data["backend_config"]["pretrained_model_name_or_path"] = (
                self.model
            )
            if self.backend:
                config_data["backend"] = self.backend
            if self.num_gpus is not None:
                config_data["num_gpus"] = self.num_gpus
            if self.target is not None:
                config_data["auto_scaling_config"]["target"] = self.target
            if self.min_instances is not None:
                config_data["auto_scaling_config"]["min_instances"] = (
                    self.min_instances
                )
            if self.max_instances is not None:
                config_data["auto_scaling_config"]["max_instances"] = (
                    self.max_instances
                )
        else:
            logger.error("You must specify either --model or --config.")
            exit(1)

        if self.enable_lora:
            config_data["backend_config"]["enable_lora"] = True
            config_data["backend_config"]["lora_adapters"] = self.lora_adapters

        self.validate_config(config_data)
        logger.info(f"Deploying model {config_data['model']}.")
        self.deploy_model(config_data)

    def deploy_model(self, config_data: dict) -> None:
        headers = {"Content-Type": "application/json"}

        if "lora_adapters" in config_data["backend_config"]:
            logger.info(
                f"Using LoRA modules: {config_data['backend_config']['lora_adapters']}"
            )

        # Send POST request to the /register endpoint
        response = requests.post(self.url, headers=headers, json=config_data)

        if response.status_code == 200:
            logger.info("Model registered successfully.")
        else:
            logger.error(
                f"Failed to register model. Status code: {response.status_code}. Response: {response.text}"
            )
