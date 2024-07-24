import json
import logging
import os
from argparse import Namespace, _SubParsersAction

import requests

from serverless_llm.cli._cli_utils import read_config, validate_config
from serverless_llm.serve.logger import init_logger

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
            "--backend", type=str, help="Overwrite the backend in the default configuration."
        )
        deploy_parser.add_argument(
            "--num_gpus", type=int, help="Overwrite the number of GPUs in the default configuration."
        )
        deploy_parser.add_argument(
            "--target", type=int, help="Overwrite the target concurrency in the default configuration."
        )
        deploy_parser.add_argument(
            "--min_instances", type=int, help="Overwrite the minimum instances in the default configuration."
        )
        deploy_parser.add_argument(
            "--max_instances", type=int, help="Overwrite the maximum instances in the default configuration."
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
        self.url = (
            os.getenv("LLM_SERVER_URL", "http://localhost:8343/") + "register"
        )
        self.default_config_path = os.path.join(
            os.path.dirname(__file__), "default_config.json"
        )

    def run(self) -> None:
        if self.config_path:
            config_data = read_config(self.config_path)
            validate_config(config_data)
            self.deploy_model(config_data)
        elif self.model:
            config_data = read_config(self.default_config_path)
            config_data["model"] = self.model
            config_data["backend_config"]["pretrained_model_name_or_path"] = (
                self.model
            )
            if self.backend is not None:
                config_data["backend"] = self.backend
            if self.num_gpus is not None:
                config_data["num_gpus"] = self.num_gpus
            if self.target is not None:
                config_data["auto_scaling_config"]["target"] = self.target
            if self.min_instances is not None:
                config_data["auto_scaling_config"]["min_instances"] = self.min_instances
            if self.max_instances is not None:
                config_data["auto_scaling_config"]["max_instances"] = self.max_instances

            logger.info(
                f"Deploying model {self.model} with custom configuration."
            )
            self.deploy_model(config_data)
        else:
            logger.error("You must specify either --model or --config.")
            exit(1)

    def deploy_model(self, config_data: dict) -> None:
        headers = {"Content-Type": "application/json"}

        # Send POST request to the /register endpoint
        response = requests.post(self.url, headers=headers, json=config_data)

        if response.status_code == 200:
            logger.info("Model registered successfully.")
        else:
            logger.error(
                f"Failed to register model. Status code: {response.status_code}. Response: {response.text}"
            )
