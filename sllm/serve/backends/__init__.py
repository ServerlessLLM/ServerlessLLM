# # ---------------------------------------------------------------------------- #
# #  serverlessllm                                                               #
# #  copyright (c) serverlessllm team 2024                                       #
# #                                                                              #
# #  licensed under the apache license, version 2.0 (the "license");             #
# #  you may not use this file except in compliance with the license.            #
# #                                                                              #
# #  you may obtain a copy of the license at                                     #
# #                                                                              #
# #                  http://www.apache.org/licenses/license-2.0                  #
# #                                                                              #
# #  unless required by applicable law or agreed to in writing, software         #
# #  distributed under the license is distributed on an "as is" basis,           #
# #  without warranties or conditions of any kind, either express or implied.    #
# #  see the license for the specific language governing permissions and         #
# #  limitations under the license.                                              #
# # ---------------------------------------------------------------------------- #
# from .dummy_backend import DummyBackend
# from .transformers_backend import TransformersBackend
# from .vllm_backend import VllmBackend
# from .sglang_backend import SGLangBackend, SGLangMode
# __all__ = ["DummyBackend", "VllmBackend", "TransformersBackend"," SGLangBackend", "SGLangMode"]
# 修改后的 __init__.py 内容
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
from .dummy_backend import DummyBackend
from .sglang_backend import SGLangBackend
from .transformers_backend import TransformersBackend
from .vllm_backend import VllmBackend

# Backend registry for easy access
backend_registry = {
    "dummy": DummyBackend,
    "sglang": SGLangBackend,
    "transformers": TransformersBackend,
    "vllm": VllmBackend,
}

__all__ = ["DummyBackend", "SGLangBackend", "VllmBackend", "TransformersBackend", "backend_registry"]
