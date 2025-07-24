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

# Adapted from https://github.com/vllm-project/vllm/blob/0ce0539d4750f9ebcd9b19d7085ca3b934b9ec67/vllm/logger.py
"""Logging configuration for sllm."""

import logging
import os
import sys
from contextvars import ContextVar
from typing import Optional

# Context variables for structured logging
correlation_id_var: ContextVar[Optional[str]] = ContextVar(
    "correlation_id", default=None
)
worker_id_var: ContextVar[Optional[str]] = ContextVar("worker_id", default=None)
model_id_var: ContextVar[Optional[str]] = ContextVar("model_id", default=None)

_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] [%(correlation_id)s|%(worker_id)s|%(model_id)s] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


class StructuredFormatter(logging.Formatter):
    """Adds logging prefix to newlines and includes context information."""

    def __init__(self, fmt, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        # Add context variables to the record
        record.correlation_id = correlation_id_var.get() or "-"
        record.worker_id = worker_id_var.get() or "-"
        record.model_id = model_id_var.get() or "-"

        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


_root_logger = logging.getLogger("sllm")
_default_handler = None


def _setup_logger():
    _root_logger.setLevel(logging.DEBUG)
    global _default_handler

    _default_handler = logging.StreamHandler(sys.stdout)
    _default_handler.flush = sys.stdout.flush  # type: ignore
    _default_handler.setLevel(logging.DEBUG)
    _root_logger.addHandler(_default_handler)

    fmt = StructuredFormatter(_FORMAT, datefmt=_DATE_FORMAT)
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


def set_worker_context(worker_id: str) -> None:
    """Set worker ID for structured logging."""
    worker_id_var.set(worker_id)


def set_model_context(model_id: str) -> None:
    """Set model ID for structured logging."""
    model_id_var.set(model_id)
