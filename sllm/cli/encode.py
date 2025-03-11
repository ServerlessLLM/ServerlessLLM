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
import json
import logging
import os
from argparse import Namespace, _SubParsersAction

import requests

from sllm.cli._cli_utils import read_config
from sllm.serve.logger import init_logger

logger = init_logger(__name__)


class EncodeCommand:
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        encode_parser = parser.add_parser(
            "encode", help="Encode using the deployed model."
        )
        encode_parser.add_argument(
            "input-path", type=str, help="Path to the JSON input file."
        )
        encode_parser.add_argument(
            "-t",
            "--threads",
            type=int,
            default=1,
            help="Number of parallel encoding processes.",
        )
        encode_parser.set_defaults(func=EncodeCommand)

    def __init__(self, args: Namespace) -> None:
        self.input_path = args.input_path
        self.threads = args.threads
        self.endpoint = "v1/embeddings"  # TODO: as a argument
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
                    executor.submit(self.encode, input_data)
                    for _ in range(self.threads)
                ]
                results = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]
                # Validate results
                if not all(results):
                    logger.error("Some encoding processes failed.")
                else:
                    logger.info(
                        f"All encoding processes done. Results: {results}"
                    )
        else:
            result = self.encode(input_data)
            logger.info(f"Embedding result: {result}")

    def encode(self, input_data: dict) -> dict:
        headers = {"Content-Type": "application/json"}

        # Send POST request to the /v1/chat/completions endpoint
        response = requests.post(self.url, headers=headers, json=input_data)

        if response.status_code == 200:
            logger.info("Encoding successful.")
            return response.json()
        else:
            logger.error(
                f"Failed to encode. Status code: {response.status_code}"
            )
            logger.error(f"Response: {response.text}")
            return None
