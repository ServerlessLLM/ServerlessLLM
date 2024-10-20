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
import json
import sys

from serverless_llm.serve.logger import init_logger

logger = init_logger(__name__)


def read_config(config_path: str) -> dict:
    """Read the JSON configuration file."""
    try:
        with open(config_path, "r") as file:
            config_data = json.load(file)
        return config_data
    except FileNotFoundError:
        logger.error(f"Config file {config_path} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from the config file {config_path}.")
        sys.exit(1)


def validate_config(config_data: dict) -> None:
    pass
    # """Validate the configuration data to ensure required fields are present."""
    # if not config_data.get("model"):
    #     logger.error("JSON file must contain 'model' field.")
    #     sys.exit(1)

    # if not config_data.get("backend"):
    #     logger.error("JSON file must contain 'backend' field.")
    #     sys.exit(1)

    # if not config_data.get("auto_scaling_config") or not isinstance(config_data.get("auto_scaling_config"), dict):
    #     logger.error("JSON file must contain 'auto_scaling_config' field as a dictionary.")
    #     sys.exit(1)

    # if config_data.get("backend") == "transformers":
    #     if not config_data.get("backend_config") or not isinstance(config_data.get("backend_config"), dict):
    #         logger.error("JSON file must contain 'backend_config' field as a dictionary.")
    #         sys.exit(1)

    #     if not config_data["backend_config"].get("pretrained_model_name_or_path"):
    #         logger.error("JSON file must contain 'pretrained_model_name_or_path' field in 'backend_config'.")
    #         sys.exit(1)
