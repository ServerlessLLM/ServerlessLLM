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

if __name__ == '__main__':
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
    parser.add_argument("sources",
                        help="Source files to hipify.",
                        nargs="*",
                        default=[])

    args = parser.parse_args()
    extra_files = [os.path.abspath(s) for s in args.sources]
    temp_dir = os.path.join(args.project_dir, "temp")
    hipify_result = hipify(project_directory=os.path.abspath(args.project_dir),
                           output_directory=os.path.abspath(temp_dir),
                           show_detailed=True,
                           is_pytorch_extension=True,
                           extra_files=extra_files,
                           )
    os.makedirs(args.output_dir, exist_ok=True)
    for _, result in hipify_result.items():
        directory, file_name = os.path.split(result.hipified_path)
        shutil.copy(result.hipified_path, os.path.join(args.output_dir, file_name))

    shutil.rmtree(temp_dir)
