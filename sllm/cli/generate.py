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
import concurrent.futures
import os
from argparse import Namespace, _SubParsersAction

import requests

from sllm.cli._cli_utils import read_config
from sllm.serve.logger import init_logger

logger = init_logger(__name__)


class GenerateCommand:
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        generate_parser = parser.add_parser(
            "generate", help="Generate using the deployed model."
        )
        generate_parser.add_argument(
            "input_path", type=str, help="Path to the JSON input file."
        )
        generate_parser.add_argument(
            "-t",
            "--threads",
            type=int,
            default=1,
            help="Number of parallel generation processes.",
        )
        generate_parser.add_argument(
            "-q",
            "--quantization",
            type=str,
            choices=["int8", "nf4", "fp4"],
            default=None,
            help="Target precision for quantization.",
        )

        generate_parser.set_defaults(func=GenerateCommand)

    def __init__(self, args: Namespace) -> None:
        self.input_path = args.input_path
        self.threads = args.threads
        self.quantization = args.quantization
        self.endpoint = "v1/chat/completions"  # TODO: as a argument
        self.url = (
            os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8343/")
            + self.endpoint
        )

    def run(self) -> None:
        input_data = read_config(self.input_path)
        if self.threads > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.threads
            ) as executor:
                futures = [
                    executor.submit(self.generate, input_data)
                    for _ in range(self.threads)
                ]
                results = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]
                # Validate results
                if not all(results):
                    logger.error("Some generations failed.")
                else:
                    logger.info(f"All generations done. Results: {results}")
        else:
            result = self.generate(input_data)
            logger.info(f"Generation result: {result}")

    def generate(self, input_data: dict) -> dict:
        if self.quantization is not None:
            input_data["quantization"] = self.quantization

        headers = {"Content-Type": "application/json"}

        # Send POST request to the /v1/chat/completions endpoint
        response = requests.post(self.url, headers=headers, json=input_data)

        if response.status_code == 200:
            logger.info("Generation successful.")
            return response.json()
        else:
            logger.error(
                f"Failed to generate. Status code: {response.status_code}"
            )
            logger.error(f"Response: {response.text}")
            return None
