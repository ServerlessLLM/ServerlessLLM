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
import argparse
import json
import os
import random
import time

import requests

# Configuration template
config_template = {
    "model": "",
    "backend": "dummy",
    "num_gpus": 1,
    "auto_scaling_config": {
        "metric": "concurrency",
        "target": 1,
        "min_instances": 0,
        "max_instances": 10,
    },
    "backend_config": {
        "pretrained_model_name_or_path": "",
        "device_map": "auto",
        "torch_dtype": "float16",
    },
}


# Function to generate random auto scaling configuration
def generate_auto_scaling_config():
    target = random.randint(1, 10)
    min_instances = 0
    max_instances = random.randint(
        10, 20
    )  # Ensure max_instances is greater than min_instances
    return {
        "metric": "concurrency",
        "target": target,
        "min_instances": min_instances,
        "max_instances": max_instances,
    }


# Function to generate configuration
def generate_config(model_name, pretrained_model_path):
    config = config_template.copy()
    config["model"] = model_name
    config["backend_config"]["pretrained_model_name_or_path"] = (
        pretrained_model_path
    )
    config["auto_scaling_config"] = generate_auto_scaling_config()
    return config


# Function to deploy model using generated configuration
def deploy_model(config, url):
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=config)

    if response.status_code == 200:
        print(f"Successfully deployed model with config: {config['model']}")
    else:
        print(f"Failed to deploy model with config: {config['model']}")
        print(response.text)


# Main script to generate and deploy models
def main(num_models, register_url):
    models = [f"dummy_model_{i}" for i in range(1, num_models + 1)]
    pretrained_paths = [f"dummy_path_{i}" for i in range(1, num_models + 1)]

    for index, (model_name, pretrained_path) in enumerate(
        zip(models, pretrained_paths), start=1
    ):
        config = generate_config(model_name, pretrained_path)
        deploy_model(config, register_url)
        time.sleep(
            1
        )  # Sleep for 1 second to avoid deploying two models at the same time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy multiple models with generated configurations."
    )
    parser.add_argument(
        "-n",
        "--num_models",
        type=int,
        required=True,
        help="Number of models to deploy.",
    )
    args = parser.parse_args()
    register_url = (
        os.getenv("LLM_SERVER_URL", "http://localhost:8343/") + "register"
    )
    main(args.num_models, register_url)
