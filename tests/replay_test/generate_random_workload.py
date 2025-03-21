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
import random


def generate_random_workload(num_models, total_requests, duration_seconds):
    workload = {f"dummy_model_{i}": [] for i in range(1, num_models + 1)}

    for _ in range(total_requests):
        model_name = random.choice(list(workload.keys()))
        timestamp = int(random.uniform(0, duration_seconds))
        workload[model_name].append(timestamp)

    # Sort timestamps for each model to simulate a real timeline of requests
    for model in workload:
        workload[model].sort()

    return workload


def main():
    parser = argparse.ArgumentParser(
        description="Generate random workload.json for sllm-cli"
    )
    parser.add_argument(
        "--num-models", type=int, required=True, help="Number of models"
    )
    parser.add_argument(
        "--request-rate",
        type=int,
        required=True,
        help="Request rate per second",
    )
    parser.add_argument(
        "--duration-minutes",
        type=int,
        required=True,
        help="Duration of the workload in minutes",
    )

    args = parser.parse_args()

    num_models = args.num_models
    request_rate = args.request_rate
    duration_minutes = args.duration_minutes
    duration_seconds = duration_minutes * 60
    total_requests = request_rate * duration_seconds

    workload = generate_random_workload(
        num_models, total_requests, duration_seconds
    )

    with open("workload.json", "w") as f:
        json.dump(workload, f, indent=4)

    print("Generated workload.json")


if __name__ == "__main__":
    main()
