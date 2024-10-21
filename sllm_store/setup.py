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
import subprocess
import sys
from pathlib import Path
from typing import Dict

from setuptools import Command, Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.command.sdist import sdist
from wheel.bdist_wheel import bdist_wheel

try:
    torch_available = True
    # The assert is not needed since Github CI does not use GPU server,
    # install cuda library is sufficient
    # assert torch.cuda.is_available() == True
    from torch.utils.cpp_extension import CUDA_HOME
except Exception:
    torch_available = False
    print(
        "[WARNING] Unable to import torch, pre-compiling ops will be disabled. "
        "Please visit https://pytorch.org/ to see how to properly install torch on your system."  # noqa: E501
    )


ROOT_DIR = os.path.dirname(__file__)


def check_nvcc_installed(cuda_home: str) -> None:
    """Check if nvcc (NVIDIA CUDA compiler) is installed."""
    try:
        _ = subprocess.check_output(
            [cuda_home + "/bin/nvcc", "-V"], universal_newlines=True
        )
    except Exception:
        raise RuntimeError(
            "nvcc is not installed or not found in your PATH. "
            "Please ensure that the CUDA toolkit is installed and nvcc is available in your PATH."  # noqa: E501
        ) from None


assert CUDA_HOME is not None, "CUDA_HOME is not set"
check_nvcc_installed(CUDA_HOME)


class BuildPackageProtos(Command):
    """Command to generate project *_pb2.py modules from proto files."""

    description = "build grpc protobuf modules"
    user_options = []

    def initialize_options(self):
        self.strict_mode = False

    def finalize_options(self):
        pass

    def _build_package_proto(self, root: str, proto_file: str) -> None:
        from grpc_tools import protoc

        command = [
            "grpc_tools.protoc",
            "-I",
            "./",
            f"--python_out={root}",
            f"--grpc_python_out={root}",
        ] + [proto_file]
        if protoc.main(command) != 0:
            raise RuntimeError("error: {} failed".format(command))

    def run(self):
        self._build_package_proto(".", "sllm_store/proto/storage.proto")


def is_ninja_available() -> bool:
    try:
        subprocess.run(["ninja", "--version"], stdout=subprocess.PIPE)
    except FileNotFoundError:
        return False
    return True


class CustomInstall(install):
    """Custom installation to ensure proto files are compiled
    and extensions are built before installation."""

    def run(self):
        self.run_command("build_ext")
        self.run_command("build_package_protos")
        super().run()


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

extras = {}

extras["test"] = [
    "pytest",
    "accelerate>=0.27.2",
    "transformers>=4.37.2",
    "parameterized",
]

sys.path.append(Path.cwd().as_posix())


class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwa) -> None:
        super().__init__(name, sources=[], **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


# Adapted from https://github.com/vllm-project/vllm/blob/a1242324c99ff8b1e29981006dfb504da198c7c3/setup.py
class cmake_build_ext(build_ext):
    did_config: Dict[str, bool] = {}

    def configure(self, ext: CMakeExtension) -> None:
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        default_cfg = "Debug" if self.debug else "Release"
        cfg = os.getenv("CMAKE_BUILD_TYPE", default_cfg)

        outdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name))
        )

        cmake_args = [
            "-DCMAKE_BUILD_TYPE={}".format(cfg),
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(outdir),
            "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={}".format(outdir),
            "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={}".format(self.build_temp),
        ]

        # verbose = bool(int(os.getenv('VERBOSE', '1')))
        verbose = True
        if verbose:
            cmake_args += ["-DCMAKE_VERBOSE_MAKEFILE=ON"]

        cmake_args += [
            "-DSLLM_STORE_PYTHON_EXECUTABLE={}".format(sys.executable)
        ]

        if is_ninja_available():
            build_tool = ["-G", "Ninja"]
            cmake_args += [
                "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
                "-DCMAKE_JOB_POOLS:STRING=compile={}".format(8),
            ]
        else:
            # Default build tool to whatever cmake picks.
            build_tool = []

        subprocess.check_call(
            ["cmake", ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp,
        )

    def build_extensions(self) -> None:
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as e:
            raise RuntimeError("Cannot find CMake executable") from e

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)

            ext_target_name = remove_prefix(ext.name, "sllm_store.")
            # num_jobs = 32

            build_args = [
                "--build",
                ".",
                "--target",
                ext_target_name,
                "-j",
                # str(num_jobs)
            ]

            subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)
            print(self.build_temp, ext_target_name)


class CustomBuild(sdist):
    """Custom build command to run build_package_protos."""

    def run(self):
        self.run_command("build_package_protos")
        super().run()


class CustomBdistWheel(bdist_wheel):
    """Custom bdist_wheel command to run build_package_protos."""

    def run(self):
        self.run_command("build_package_protos")
        super().run()


cmdclass = {
    "build_ext": cmake_build_ext,
    "build_package_protos": BuildPackageProtos,
    "install": CustomInstall,
    "sdist": CustomBuild,
    "bdist_wheel": CustomBdistWheel,
}

setup(
    name="serverless-llm-store",
    version="0.5.0",
    ext_modules=[
        CMakeExtension(name="sllm_store._C"),
        CMakeExtension(name="sllm_store.sllm_store_server"),
    ],
    install_requires=install_requires,
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    extras_require=extras,
    entry_points={
        "console_scripts": ["sllm-store-server=sllm_store.server.server:main"],
    },
    package_data={
        "sllm_store": ["py.typed", "*.so", "sllm_store_server"],
    },
    packages=[
        "sllm_store",
        "sllm_store.proto",
        "sllm_store.server",
    ],
    cmdclass=cmdclass,
)
