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
import asyncio

from serverless_llm.cli.delete import DeleteCommand
from serverless_llm.cli.deploy import DeployCommand
from serverless_llm.cli.generate import GenerateCommand
from serverless_llm.cli.encode import EncodeCommand
from serverless_llm.cli.replay import ReplayCommand
from serverless_llm.cli.update import UpdateCommand
from serverless_llm.serve.logger import init_logger

logger = init_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        "sllm-cli", usage="sllm-cli <command> [<args>]"
    )
    commands_parser = parser.add_subparsers(help="sllm-cli command helpers")

    # Register commands
    DeployCommand.register_subcommand(commands_parser)
    GenerateCommand.register_subcommand(commands_parser)
    EncodeCommand.register_subcommand(commands_parser)
    ReplayCommand.register_subcommand(commands_parser)
    DeleteCommand.register_subcommand(commands_parser)
    UpdateCommand.register_subcommand(commands_parser)

    # Let's go
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    service = args.func(args)
    if asyncio.iscoroutinefunction(service.run):
        asyncio.run(service.run())
    else:
        service.run()


if __name__ == "__main__":
    main()
