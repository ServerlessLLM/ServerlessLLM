import asyncio

import grpc

from sllm_store.proto import storage_pb2, storage_pb2_grpc
from sllm_store.logger import init_logger

logger = init_logger(__name__)


class StorageServicer(storage_pb2_grpc.StorageServicer):
    # NOTE: PLACEHOLDER IMPLEMENTATION
    async def LoadModelAsync(self, request, context):
        logger.info(
            f"LoadModelAsync: {request.model_path}, {request.replica_uuid}"
        )
        return storage_pb2.LoadModelResponse(model_path=request.model_path)

    async def ConfirmModel(self, request, context):
        logger.info(
            f"ConfirmModel: {request.model_path}, {request.replica_uuid}"
        )
        return storage_pb2.ConfirmModelResponse(model_path=request.model_path)

    async def UnloadModel(self, request, context):
        logger.info(
            f"UnloadModel: {request.model_path}, {request.replica_uuid}"
        )
        return storage_pb2.UnloadModelResponse(model_path=request.model_path)

    async def ClearMem(self, request, context):
        logger.info("ClearMem")
        return storage_pb2.ClearMemResponse()

    async def RegisterModel(self, request, context):
        logger.info(f"RegisterModel: {request.model_path}")
        return storage_pb2.RegisterModelResponse(
            model_path=request.model_path, model_size=1024
        )

    async def GetServerConfig(self, request, context):
        logger.info("GetServerConfig")
        return storage_pb2.GetServerConfigResponse(
            mem_pool_size=1024, chunk_size=128
        )


async def serve(host="0.0.0.0", port=50051):
    server = grpc.aio.server()
    storage_pb2_grpc.add_StorageServicer_to_server(StorageServicer(), server)
    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    logger.info(f"Starting gRPC server on {listen_addr}")
    await server.start()

    # Graceful shutdown handling
    try:
        # Block until cancelled
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down gRPC server")
        await server.stop(5)  # Graceful shutdown with a 5-second timeout
