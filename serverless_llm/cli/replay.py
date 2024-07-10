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
import asyncio
import json
import logging
import os
import time
from argparse import Namespace, _SubParsersAction

from openai import AsyncOpenAI

from serverless_llm.cli._cli_utils import read_config
from serverless_llm.serve.logger import init_logger

logger = init_logger(__name__)


class ReplayCommand:
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        replay_parser = parser.add_parser(
            "replay", help="Replay requests based on workload and dataset."
        )
        replay_parser.add_argument(
            "--workload",
            type=str,
            required=True,
            help="Path to the JSON workload file.",
        )
        replay_parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            help="Path to the JSON dataset file.",
        )
        replay_parser.add_argument(
            "--output",
            type=str,
            default="latency_results.json",
            help="Path to the output JSON file for latency results.",
        )
        replay_parser.set_defaults(func=ReplayCommand)

    def __init__(self, args: Namespace) -> None:
        self.workload_path = args.workload
        self.dataset_path = args.dataset
        self.output_path = args.output
        self.url = os.getenv("LLM_SERVER_URL", "http://localhost:8343/")
        self.base_url = self.url + "v1/"

        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key="API_KEY_PLACEHOLDER",  # Placeholder for API key
        )
        self.latency_results = []

    async def run(self) -> None:
        workload = read_config(self.workload_path)
        dataset = read_config(self.dataset_path)

        input_texts = dataset.get("input_text", [])
        output_lengths = dataset.get("output_length", [])

        if not input_texts or not output_lengths:
            logger.error(
                "Dataset is missing required fields: input_text and output_length"
            )
            return

        tasks = []
        for model_name, times in workload.items():
            for i, time_offset in enumerate(times):
                if i >= len(input_texts) or i >= len(output_lengths):
                    logger.error(f"Index {i} is out of bounds for the dataset")
                    break
                input_text = input_texts[i]
                output_length = output_lengths[i]
                tasks.append(
                    self.schedule_request(
                        model_name, input_text, output_length, time_offset
                    )
                )

        await asyncio.gather(*tasks)
        self.write_latency_results()

    async def schedule_request(
        self,
        model_name: str,
        input_text: str,
        output_length: int,
        time_offset: float,
    ) -> None:
        await asyncio.sleep(time_offset)
        request_data = {
            "model": model_name,
            "messages": [{"role": "user", "content": input_text}],
            "max_tokens": output_length,
        }
        await self.send_request(request_data)

    async def send_request(self, request_data: dict) -> None:
        start_time = time.time()
        try:
            response = await self.client.chat.completions.create(**request_data)
            end_time = time.time()
            latency = end_time - start_time
            logger.info(
                f"Generation successful: {response.choices[0].message.content}"
            )
            self.latency_results.append(
                {
                    "model": request_data["model"],
                    "input_text": request_data["messages"][0]["content"],
                    "latency": latency,
                }
            )
            return response
        except Exception as e:
            end_time = time.time()
            latency = end_time - start_time
            logger.error(f"Failed to generate. Error: {str(e)}")
            self.latency_results.append(
                {
                    "model": request_data["model"],
                    "input_text": request_data["messages"][0]["content"],
                    "latency": latency,
                    "error": str(e),
                }
            )

    def write_latency_results(self) -> None:
        with open(self.output_path, "w") as f:
            json.dump(self.latency_results, f, indent=4)
        logger.info(f"Latency results written to {self.output_path}")
