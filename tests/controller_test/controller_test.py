# ---------------------------------------------------#
# before running this test, please make sure that   #
# the ray cluster and sllm-store-server is running. #
# ---------------------------------------------------#

import pytest
import pytest_asyncio
import logging
import requests
import pprint

from typing import List, Mapping, Optional

import ray
from ray.util import inspect_serializability
from ray.cluster_utils import Cluster
from serverless_llm.serve.controller import SllmController

from unittest.mock import patch, Mock, MagicMock, AsyncMock
from mock_store_manager import MockStoreManagerBadNetwork, MockStoreManagerGoodNetwork

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def setup_environment():
    # start up a mock cluster
    mock_cluster = Cluster(
        initialize_head=True,
        head_node_args={"num_cpus": 4, "resources": {"control_node": 1}},
    )

    ray.init(address=mock_cluster.address)
    # ray.init()

    yield mock_cluster
    # yield None

    ray.shutdown()
    # mock_cluster.shutdown()


@pytest.fixture(scope="session")
def setup_controller(setup_environment):
    cluster = setup_environment
    cluster.add_node(num_cpus=4, num_gpus=1, object_store_memory=75 *
                     1024 * 1024, resources={"worker_node": 1, "worker_id_0": 1})
    controller = SllmController({"hardware_config": None}, debug=True)

    yield controller


@pytest.mark.asyncio
@patch("serverless_llm.serve.controller.StoreManager", new=MockStoreManagerBadNetwork)
async def test_network_failed(setup_controller):
    controller = setup_controller
    with patch("serverless_llm.serve.controller.StoreManager", new=MockStoreManagerBadNetwork):
        await controller.start()
        try:
            # controller = ray.get_actor("controller")
            await controller.register(
                {
                    "model": "facebook/opt-1.3b",
                    "backend": "vllm",
                    "num_gpus": 1,
                    "auto_scaling_config": {
                        "metric": "concurrency",
                        "target": 1,
                        "min_instances": 0,
                        "max_instances": 10
                    },
                    "backend_config": {
                        "pretrained_model_name_or_path": "facebook/opt-1.3b",
                        "device_map": "auto",
                        "torch_dtype": "float16"
                    }
                }
            )
            assert False
        except RuntimeError as e:
            assert "Mock network error" in str(e), f"Got unexpected exception {e}"
            assert len(controller.registered_models) == 0, f"Got unexpected registered models {controller.registered_models}"

