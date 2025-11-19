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

# Lazy imports to avoid loading heavy dependencies on head nodes
__all__ = [
    "DummyBackend",
    "VllmBackend",
    "TransformersBackend",
    "VllmMoeCapBackend",
]


def __getattr__(name):
    """Lazy import backends only when accessed."""
    if name == "DummyBackend":
        from .dummy_backend import DummyBackend

        return DummyBackend
    elif name == "VllmBackend":
        from .vllm_backend import VllmBackend

        return VllmBackend
    elif name == "TransformersBackend":
        from .transformers_backend import TransformersBackend

        return TransformersBackend
    elif name == "VllmMoeCapBackend":
        from .vllm_moecap_backend import VllmMoeCapBackend

        return VllmMoeCapBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
