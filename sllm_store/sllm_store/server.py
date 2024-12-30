import asyncio
import grpc
from sllm_store.proto import storage_pb2, storage_pb2_grpc
from sllm_store.logger import init_logger
from sllm_store._checkpoint_store import (
    CheckpointStore,
    MemCopyChunk,
    MemCopyHandle,
)

logger = init_logger(__name__)


class StorageServicer(storage_pb2_grpc.StorageServicer):
    def __init__(
        self,
        storage_path,
        mem_pool_size,
        num_thread,
        chunk_size,
        registration_required,
    ):
        if not storage_path:
            logger.error("storage_path is empty")
            raise ValueError("storage_path is empty")

        if mem_pool_size <= 0:
            logger.error("mem_pool_size must be greater than 0")
            raise ValueError("Invalid mem_pool_size")

        self.storage = CheckpointStore(
            storage_path, mem_pool_size, num_thread, chunk_size
        )
        self.registration_required = registration_required

    async def LoadModelAsync(self, request, context):
        model_path = request.model_path
        if not model_path:
            logger.error("model_path is empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return storage_pb2.LoadModelResponse()

        if not self.registration_required:
            model_size = self.storage.register_model_info(model_path)
            if model_size < 0:
                logger.error("RegisterModel failed")
                context.set_code(grpc.StatusCode.INTERNAL)
                return storage_pb2.LoadModelResponse()

        device_type = request.target_device_type
        if device_type == storage_pb2.DEVICE_TYPE_CPU:
            ret = self.storage.load_model_from_disk_async(model_path)
        elif device_type == storage_pb2.DEVICE_TYPE_GPU:
            replica_uuid = request.replica_uuid
            if not replica_uuid:
                logger.error("replica_uuid is empty")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return storage_pb2.LoadModelResponse()

            gpu_memory_handles = {
                device_uuid: [
                    MemCopyHandle(cuda_ipc_handle=handle.cuda_ipc_handle)
                    for handle in handle_list.handles
                ]
                for device_uuid, handle_list in request.handles.items()
            }
            mem_copy_chunks = {
                device_uuid: [
                    MemCopyChunk(
                        src_offset=chunk.src_offset,
                        size=chunk.size,
                        dst_offset=chunk.dst_offset,
                        handle_idx=chunk.handle_idx,
                    )
                    for chunk in chunk_list.chunks
                ]
                for device_uuid, chunk_list in request.chunks.items()
            }
            ret = self.storage.load_model_from_mem_async(
                model_path, replica_uuid, gpu_memory_handles, mem_copy_chunks
            )
        else:
            logger.error(f"Unsupported device type: {device_type}")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return storage_pb2.LoadModelResponse()

        if ret != 0:
            logger.error("LoadModel failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            return storage_pb2.LoadModelResponse()

        logger.info(
            f"LoadModel: success {model_path} with target {device_type}"
        )
        return storage_pb2.LoadModelResponse(model_path=model_path)

    async def ConfirmModel(self, request, context):
        model_path = request.model_path
        replica_uuid = request.replica_uuid
        device_type = request.target_device_type

        if not model_path:
            logger.error("model_path is empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return storage_pb2.ConfirmModelResponse()

        if device_type != storage_pb2.DEVICE_TYPE_GPU:
            logger.error(f"Unsupported device type: {device_type}")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return storage_pb2.ConfirmModelResponse()

        for i in range(5):
            ret = self.storage.wait_model_in_gpu(model_path, replica_uuid)
            if ret == 0:
                logger.info(
                    f"Confirm model {model_path} replica {replica_uuid} success"
                )
                return storage_pb2.ConfirmModelResponse(model_path=model_path)
            logger.info(f"Confirm model failed, retry {i + 1}")

            await asyncio.sleep(0.05)

        logger.error(
            f"Confirm model {model_path} replica {replica_uuid} failed"
        )
        context.set_code(grpc.StatusCode.INTERNAL)
        return storage_pb2.ConfirmModelResponse()

    async def UnloadModel(self, request, context):
        model_path = request.model_path
        device_type = request.target_device_type

        if not model_path:
            logger.error("model_path is empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return storage_pb2.UnloadModelResponse()

        if device_type != storage_pb2.DEVICE_TYPE_CPU:
            logger.error(f"Unsupported device type: {device_type}")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return storage_pb2.UnloadModelResponse()

        for i in range(5):
            ret = self.storage.unload_model_from_host(model_path)
            if ret == 0:
                logger.info(f"UnloadModel: success {model_path}")
                return storage_pb2.UnloadModelResponse(model_path=model_path)
            logger.info(f"UnloadModel failed, retry {i + 1}")

            await asyncio.sleep(0.01)

        logger.error(f"UnloadModel failed for model {model_path}")
        context.set_code(grpc.StatusCode.INTERNAL)
        return storage_pb2.UnloadModelResponse()

    async def ClearMem(self, request, context):
        ret = self.storage.clear_mem()
        if ret != 0:
            logger.error("ClearMem failed")
            context.set_code(grpc.StatusCode.INTERNAL)
        return storage_pb2.ClearMemResponse()

    async def RegisterModel(self, request, context):
        model_path = request.model_path
        if not model_path:
            logger.error("model_path is empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return storage_pb2.RegisterModelResponse()

        model_size = self.storage.register_model_info(model_path)
        if model_size < 0:
            logger.error("RegisterModel failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            return storage_pb2.RegisterModelResponse()

        return storage_pb2.RegisterModelResponse(
            model_path=model_path, model_size=model_size
        )

    async def GetServerConfig(self, request, context):
        return storage_pb2.GetServerConfigResponse(
            mem_pool_size=self.storage.get_mem_pool_size(),
            chunk_size=self.storage.get_chunk_size(),
        )


async def serve(host="0.0.0.0", port=50051, **kwargs):
    server = grpc.aio.server()
    storage_pb2_grpc.add_StorageServicer_to_server(
        StorageServicer(**kwargs), server
    )
    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    logger.info(f"Starting gRPC server on {listen_addr}")
    await server.start()

    try:
        await server.wait_for_termination()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down gRPC server")
        await server.stop(5)
