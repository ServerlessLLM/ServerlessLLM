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
import argparse
import json
import logging
import re
import sys

import ray
import uvicorn

from serverless_llm.serve.app_lib import create_app
from serverless_llm.serve.controller import SllmController
from serverless_llm.serve.logger import init_logger

logger = init_logger(__name__)


def process_hardware_config(hardware_config):
    # Define conversion factors
    conversion_factors = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
        "B/s": 1,
        "KB/s": 1024,
        "MB/s": 1024**2,
        "GB/s": 1024**3,
        "TB/s": 1024**4,
        "bps": 1 / 8,  # bits per second to bytes per second
        "Kbps": 1024 / 8,
        "Mbps": (1024**2) / 8,
        "Gbps": (1024**3) / 8,
        "Tbps": (1024**4) / 8,
    }

    def convert_value(value):
        # Regular expression to match number and unit
        pattern = re.compile(r"([\d.]+)([a-zA-Z/]+)")
        match = pattern.fullmatch(value.strip())
        if not match:
            raise ValueError(f"Invalid value format: {value}")

        number, unit = match.groups()
        number = float(number)

        if unit not in conversion_factors:
            raise ValueError(f"Unknown unit in value: {value}")

        return number * conversion_factors[unit]

    converted_config = {}

    for key, specs in hardware_config.items():
        converted_specs = {}
        for spec_name, spec_value in specs.items():
            converted_specs[spec_name] = convert_value(spec_value)
        converted_config[key] = converted_specs

    return converted_config


def main():
    parser = argparse.ArgumentParser(
        description="ServerlessLLM CLI for model management."
    )
    subparsers = parser.add_subparsers(dest="command")

    start_parser = subparsers.add_parser("start", help="Start the Sllm server.")
    start_parser.add_argument(
        "--host",
        default="0.0.0.0",
        type=str,
        help="Host IP to run the server on.",
    )
    start_parser.add_argument(
        "--port", default=8343, type=int, help="Port to run the server on."
    )
    start_parser.add_argument(
        "--hardware-config",
        default=None,
        type=str,
        help="Path to hardware config file.",
    )
    args = parser.parse_args()

    try:
        if args.command == "start":
            app = create_app()
            hardware_config_path = args.hardware_config
            hardware_config = None
            if hardware_config_path is not None:
                with open(hardware_config_path, "r") as f:
                    hardware_config = json.load(f)
                hardware_config = process_hardware_config(hardware_config)
            controller_cls = ray.remote(SllmController)
            controller = controller_cls.options(name="controller", resources={"control_node": 0.1}).remote(
                {"hardware_config": hardware_config}
            )
            ray.get(controller.start.remote())

            uvicorn.run(app, host=args.host, port=args.port)
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        ray.get(controller.shutdown.remote())


if __name__ == "__main__":
    main()
