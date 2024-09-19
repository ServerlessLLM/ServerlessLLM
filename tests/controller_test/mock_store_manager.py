from unittest.mock import AsyncMock, MagicMock
import logging
from serverless_llm.serve.store_manager import StoreManager

logger = logging.getLogger(__name__)

class MockStoreManagerBadNetwork(StoreManager):

    async def register(self, model_config):
        logger.debug(f"Registering model {model_config} with bad network")
        raise RuntimeError("Mock network error")


class MockStoreManagerGoodNetwork(StoreManager):

    async def register(self, model_config):
        logger.debug(f"Registering model {model_config} with bad network")
        # raise RuntimeError("Mock network error")
        return

