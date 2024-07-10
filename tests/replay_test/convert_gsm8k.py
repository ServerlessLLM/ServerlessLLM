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
import json

from datasets import load_dataset


def convert_gsm8k_to_custom_format(config="main", split="train"):
    # Load the gsm8k dataset with the specified config
    dataset = load_dataset("gsm8k", config, split=split)

    input_texts = []
    output_lengths = []

    # Extract questions and answers, and compute output lengths
    for example in dataset:
        question = example["question"]
        answer = example["answer"]
        input_texts.append(question)
        output_lengths.append(calculate_tokens(answer))

    # Create the final format
    formatted_data = {
        "input_text": input_texts,
        "output_length": output_lengths,
    }

    return formatted_data


def calculate_tokens(answer):
    return len(
        answer.split()
    )  # TODO: undecided on how to calculate output tokens


def main():
    # Convert the dataset
    formatted_data = convert_gsm8k_to_custom_format()

    # Save to a JSON file
    with open("gsm8k_dataset.json", "w") as f:
        json.dump(formatted_data, f, indent=4)

    print("Converted dataset saved as gsm8k_dataset.json")


if __name__ == "__main__":
    main()
