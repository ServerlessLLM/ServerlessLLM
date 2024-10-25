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

# Adapted from https://github.com/vllm-project/vllm/blob/0ce0539d4750f9ebcd9b19d7085ca3b934b9ec67/vllm/logger.py
"""Logging configuration for sllm_store."""

import logging
import os
import sys

_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


_root_logger = logging.getLogger("sllm_store")
_default_handler = None


def _setup_logger():
    _root_logger.setLevel(logging.DEBUG)
    global _default_handler

    _default_handler = logging.StreamHandler(sys.stdout)
    _default_handler.flush = sys.stdout.flush  # type: ignore
    _default_handler.setLevel(logging.DEBUG)
    _root_logger.addHandler(_default_handler)

    fmt = NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT)
    _default_handler.setFormatter(fmt)
    # Setting this will avoid the message
    # being propagated to the parent logger.
    _root_logger.propagate = False


_setup_logger()


def init_logger(name: str):
    # Use the same settings as above for root logger
    logger = logging.getLogger(name)

    log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()
    logger.setLevel(log_level)

    # Ensure the handler's level matches the logger's level
    if _default_handler:
        _default_handler.setLevel(log_level)

    logger.addHandler(_default_handler)
    logger.propagate = False

    return logger
