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
import io
import os
import sys
from pathlib import Path

from setuptools import setup

ROOT_DIR = os.path.dirname(__file__)


def fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""


install_requires = fetch_requirements("requirements.txt")
install_requires_worker = fetch_requirements("requirements-worker.txt")

extras = {}
extras["test"] = install_requires_worker + ["pytest", "pytest-asyncio"]
extras["worker"] = install_requires_worker


sys.path.append(Path.cwd().as_posix())


setup(
    name="serverless-llm",
    version="0.0.1.dev1",
    install_requires=install_requires,
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    extras_require=extras,
    entry_points={
        "console_scripts": [
            "sllm-cli=serverless_llm.cli.sllm_cli:main",
            "sllm-serve=serverless_llm.serve.commands.serve.sllm_serve:main",
        ],
    },
    include_package_data=True,
    package_data={
        "serverless_llm.serve": ["py.typed", "serverless_llm.sllm_serve"],
        "serverless_llm.cli": ["default_config.json"],
    },
    packages=[
        "serverless_llm.serve",
        "serverless_llm.cli",
        "serverless_llm.serve.commands.serve",
        "serverless_llm.serve.backends",
        "serverless_llm.serve.routers",
        "serverless_llm.serve.schedulers",
    ],
)
