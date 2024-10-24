#!/usr/bin/env python3

#
# A command line tool for running pytorch's hipify preprocessor on CUDA
# source files.
#
# See https://github.com/ROCm/hipify_torch
# and <torch install dir>/utils/hipify/hipify_python.py
#

import argparse
import os
import shutil

from torch.utils.hipify.hipify_python import hipify


def vllm_hipify(args):
    # copied from vllm's cmake/hipify.py
    # Limit include scope to project_dir only
    includes = [os.path.join(args.project_dir, "*")]

    # Get absolute path for all source files.
    extra_files = [os.path.abspath(s) for s in args.sources]

    # Copy sources from project directory to output directory.
    # The directory might already exist to hold object files so we ignore that.
    shutil.copytree(args.project_dir, args.output_dir, dirs_exist_ok=True)

    hipify_result = hipify(
        project_directory=args.project_dir,
        output_directory=args.output_dir,
        header_include_dirs=[],
        includes=includes,
        extra_files=extra_files,
        show_detailed=True,
        is_pytorch_extension=True,
        hipify_extra_files_only=True,
    )

    hipified_sources = []
    for source in args.sources:
        s_abs = os.path.abspath(source)
        hipified_s_abs = (
            hipify_result[s_abs].hipified_path
            if (
                s_abs in hipify_result
                and hipify_result[s_abs].hipified_path is not None
            )
            else s_abs
        )
        hipified_sources.append(hipified_s_abs)

    assert len(hipified_sources) == len(args.sources)

    # Print hipified source files.
    print("\n".join(hipified_sources))


def sllm_hipify(args):
    extra_files = [os.path.abspath(s) for s in args.sources]
    temp_dir = os.path.join(args.project_dir, "temp")
    hipify_result = hipify(
        project_directory=os.path.abspath(args.project_dir),
        output_directory=os.path.abspath(temp_dir),
        show_detailed=True,
        is_pytorch_extension=True,
        extra_files=extra_files,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    for _, result in hipify_result.items():
        directory, file_name = os.path.split(result.hipified_path)
        shutil.copy(
            result.hipified_path, os.path.join(args.output_dir, file_name)
        )

    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Project directory where all the source + include files live.
    parser.add_argument(
        "-p",
        "--project_dir",
        help="The project directory.",
    )

    # Directory where hipified files are written.
    parser.add_argument(
        "-o",
        "--output_dir",
        help="The output directory.",
    )

    # Source files to convert.
    parser.add_argument(
        "sources", help="Source files to hipify.", nargs="*", default=[]
    )

    # vLLM mode or not
    parser.add_argument(
        "-s", "--sllm", help="hipify for sllm", action="store_true"
    )

    args = parser.parse_args()
    if args.sllm:
        sllm_hipify(args)
    else:
        vllm_hipify(args)
