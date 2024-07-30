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
import threading

import grpc
import serverless_llm_store.proto.storage_pb2 as storage_pb2
import serverless_llm_store.proto.storage_pb2_grpc as storage_pb2_grpc
from serverless_llm_store.logger import init_logger

logger = init_logger(__name__)


# This is a singleton class that manages the checkpoint
class SllmStoreClient:
    def __init__(self, server_address="localhost:8073"):
        self.server_address = server_address
        self.channel = grpc.insecure_channel(server_address)
        self.stub = storage_pb2_grpc.StorageStub(self.channel)
        self.checkpoints_in_gpu = {}

    def __del__(self):
        # TODO: cleanup
        pass

    def load_into_cpu(self, model_name):
        request = storage_pb2.LoadModelRequest(
            model_name=model_name,
            target_device_type=storage_pb2.DeviceType.DEVICE_TYPE_CPU,
        )
        try:
            response = self.stub.LoadModelAsync(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.CANCELLED:
                logger.error(f"Model not loaded {e}")
                return False
            else:
                logger.error(f"Error: {e}")
                return False
        else:
            return response

    def unload_from_cpu(self, model_name):
        request = storage_pb2.UnloadModelRequest(
            model_name=model_name,
            target_device_type=storage_pb2.DeviceType.DEVICE_TYPE_CPU,
        )
        try:
            response = self.stub.UnloadModel(request)
        except grpc.RpcError as e:
            logger.error(f"Error: {e}")
            return False
        else:
            return response

    def load_into_gpu(
        self, model_name, replica_uuid, tensor_copy_chunks, cuda_memory_handles
    ):
        logger.debug(f"load_into_gpu: {model_name}, {replica_uuid}")

        gpu_chunk_map = {}
        for device_uuid, chunks in tensor_copy_chunks.items():
            gpu_chunk_map[device_uuid] = storage_pb2.MemCopyChunkList(
                chunks=[
                    storage_pb2.MemCopyChunk(
                        src_offset=chunk[0],
                        size=chunk[1],
                        dst_offset=chunk[2],
                        handle_idx=chunk[3],
                    )
                    for chunk in chunks
                ]
            )
        cuda_handle_map = {}
        for device_uuid, handles in cuda_memory_handles.items():
            cuda_handle_map[device_uuid] = storage_pb2.MemCopyHandleList(
                handles=[
                    storage_pb2.MemCopyHandle(
                        cuda_ipc_handle=handle_str,
                    )
                    for handle_str in handles
                ]
            )
        request = storage_pb2.LoadModelRequest(
            model_name=model_name,
            replica_uuid=replica_uuid,
            chunks=gpu_chunk_map,
            handles=cuda_handle_map,
            target_device_type=storage_pb2.DeviceType.DEVICE_TYPE_GPU,
        )
        try:
            response = self.stub.LoadModelAsync(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.CANCELLED:
                logger.error(f"Model not loaded {e}")
            else:
                logger.error(f"Error: {e}")
            return False
        else:
            logger.info(f"Model loaded: {model_name}, {replica_uuid}")
            return response

    def confirm_model_loaded(self, model_name, replica_uuid):
        logger.info(f"confirm_model_loaded: {model_name}, {replica_uuid}")
        request = storage_pb2.ConfirmModelRequest(
            model_name=model_name,
            replica_uuid=replica_uuid,
            target_device_type=storage_pb2.DeviceType.DEVICE_TYPE_GPU,
        )
        try:
            response = self.stub.ConfirmModel(request)
            logger.info(f"Model loaded")
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.CANCELLED:
                logger.error("Model not loaded")
                return False
            else:
                logger.error(f"Error: {e}")
                return False

    def register_model(self, model_name) -> int:
        logger.info(f"register_model: {model_name}")
        request = storage_pb2.RegisterModelRequest(model_name=model_name)
        try:
            response = self.stub.RegisterModel(request)
        except grpc.RpcError as e:
            logger.error(f"Error: {e}")
            return -1
        else:
            logger.info(f"Model registered")
            return response.model_size

    def get_server_config(self):
        request = storage_pb2.GetServerConfigRequest()
        try:
            response = self.stub.GetServerConfig(request)
        except grpc.RpcError as e:
            logger.error(f"Error: {e}")
            return None
        else:
            return {"chunk_size": response.chunk_size}
