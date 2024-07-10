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
import asyncio
import time

import pytest
import ray

from serverless_llm.serve.backends.vllm_backend_v2 import VllmBackend

backend_config = {
    "pretrained_model_name_or_path": "facebook/opt-125m",
    "gpu_memory_utilization": 0.3,
    "trace_debug": True,
}

request_data = {
    "model": "opt-125m",
    "messages": [{"role": "user", "content": "Please introduce yourself."}],
    "temperature": 0.3,
    "max_tokens": 50,
    "min_tokens": 10,
}

long_prompt = (
    "You are a helpful assistant in recognizes the content of tables in markdown format. Here is a table as follows.\n# Table\n"
    + """
| ID  | Name          | Age | Occupation    | Country       | Email                  | Phone Number   | Address                       |
|-----|---------------|-----|---------------|---------------|------------------------|----------------|------------------------------|
| 1   | John Doe      | 29  | Engineer      | USA           | john.doe@example.com   | 555-1234       | 123 Elm St, Springfield, IL  |
| 2   | Jane Smith    | 34  | Doctor        | Canada        | jane.smith@example.com | 555-5678       | 456 Oak St, Toronto, ON      |
| 3   | Alice Johnson | 27  | Teacher       | UK            | alice.j@example.com    | 555-8765       | 789 Pine St, London, UK      |
| 4   | Bob Brown     | 45  | Artist        | Australia     | bob.b@example.com      | 555-4321       | 321 Maple St, Sydney, NSW    |
| 5   | Carol White   | 31  | Scientist     | New Zealand   | carol.w@example.com    | 555-6789       | 654 Birch St, Wellington, NZ |
| 6   | Dave Green    | 28  | Lawyer        | Ireland       | dave.g@example.com     | 555-3456       | 987 Cedar St, Dublin, IE     |
| 7   | Emma Black    | 40  | Musician      | USA           | emma.b@example.com     | 555-1111       | 246 Ash St, New York, NY     |
| 8   | Frank Blue    | 37  | Chef          | Canada        | frank.b@example.com    | 555-2222       | 135 Spruce St, Vancouver, BC |
| 9   | Grace Yellow  | 50  | Engineer      | UK            | grace.y@example.com    | 555-3333       | 864 Fir St, Manchester, UK   |
| 10  | Henry Violet  | 32  | Artist        | Australia     | henry.v@example.com    | 555-4444       | 753 Willow St, Melbourne, VIC|
| 11  | Irene Orange  | 26  | Scientist     | New Zealand   | irene.o@example.com    | 555-5555       | 912 Poplar St, Auckland, NZ  |
| 12  | Jack Indigo   | 38  | Teacher       | Ireland       | jack.i@example.com     | 555-6666       | 159 Elm St, Cork, IE         |
| 13  | Karen Red     | 41  | Lawyer        | USA           | karen.r@example.com    | 555-7777       | 357 Cedar St, Boston, MA     |
| 14  | Leo Brown     | 30  | Chef          | Canada        | leo.b@example.com      | 555-8888       | 246 Oak St, Calgary, AB      |
| 15  | Mia Green     | 33  | Musician      | UK            | mia.g@example.com      | 555-9999       | 975 Pine St, Edinburgh, UK   |
| 16  | Noah Yellow   | 29  | Doctor        | Australia     | noah.y@example.com     | 555-0000       | 864 Birch St, Brisbane, QLD  |
| 17  | Olivia Blue   | 35  | Engineer      | New Zealand   | olivia.b@example.com   | 555-1212       | 753 Maple St, Hamilton, NZ   |
| 18  | Peter Black   | 42  | Artist        | Ireland       | peter.b@example.com    | 555-3434       | 912 Fir St, Limerick, IE     |
| 19  | Quinn White   | 28  | Scientist     | USA           | quinn.w@example.com    | 555-5656       | 159 Willow St, Seattle, WA   |
| 20  | Rachel Red    | 31  | Teacher       | Canada        | rachel.r@example.com   | 555-7878       | 357 Poplar St, Ottawa, ON    |
| 21  | Steve Green   | 44  | Lawyer        | UK            | steve.g@example.com    | 555-9090       | 753 Elm St, Birmingham, UK   |
| 22  | Tina Blue     | 36  | Musician      | Australia     | tina.b@example.com     | 555-1213       | 864 Cedar St, Perth, WA      |
| 23  | Umar Black    | 39  | Chef          | New Zealand   | umar.b@example.com     | 555-3435       | 975 Spruce St, Christchurch, NZ|
| 24  | Victor Yellow | 43  | Engineer      | Ireland       | victor.y@example.com   | 555-5657       | 246 Willow St, Galway, IE    |
| 25  | Wendy Orange  | 27  | Artist        | USA           | wendy.o@example.com    | 555-7879       | 135 Elm St, Denver, CO       |
"""
)


@pytest.fixture(scope="module")
def ray_start():
    ray.init(num_gpus=1)
    yield None
    ray.shutdown()


@pytest.fixture()
def vllm_backend(ray_start):
    backend = VllmBackend.options(num_gpus=1).remote(
        backend_config=backend_config
    )

    yield backend

    backend.shutdown.remote()

    del backend


@pytest.mark.asyncio
async def test_generate(vllm_backend):
    result = await vllm_backend.generate.remote(request_data=request_data)
    print(f"generated result: {result}")

    assert result["usage"]["completion_tokens"] >= request_data["min_tokens"]
    assert result["usage"]["completion_tokens"] <= request_data["max_tokens"]


@pytest.mark.asyncio
async def test_get_current_tokens(vllm_backend):
    new_request = {
        "model": "opt-125m",
        "messages": [{"role": "user", "content": long_prompt[:1024]}],
        "temperature": 0.3,
        "max_tokens": 1000,
        "min_tokens": 1000,
    }

    async def generate_request(request_data):
        return await vllm_backend.generate.remote(request_data=request_data)

    result = await generate_request(new_request)
    tokens = await vllm_backend.get_current_tokens.remote()
    assert len(tokens) > 0
    print(f"tokens: {tokens}")


@pytest.mark.asyncio
async def test_batch_processing(ray_start):
    request_num = 10
    request_list: list[dict] = []
    for i in range(request_num):
        new_request = {
            "model": "opt-125m",
            "messages": [{"role": "user", "content": f"request {i}"}],
            "temperature": 0.3,
            "max_tokens": 1000,
            "min_tokens": 1000,
        }
        request_list.append(new_request)

    async def generate_request(backend, request_data):
        return await backend.generate.remote(request_data=request_data)

    # batch process time cost:
    start_time = time.time()
    batch_backend = VllmBackend.options(num_gpus=1).remote(
        backend_config=backend_config
    )
    tasks = [
        asyncio.create_task(
            generate_request(backend=batch_backend, request_data=request)
        )
        for request in request_list
    ]
    # make sure all requests are sent
    created_tasks = await asyncio.gather(*tasks)
    batch_time = time.time() - start_time
    tokens = await batch_backend.get_current_tokens.remote()
    assert len(tokens) == request_num
    print(f"current tokens: {tokens}")
    batch_backend.shutdown.remote()

    del batch_backend
    # single process time cost:
    start_time = time.time()

    normal_backend = VllmBackend.options(num_gpus=1).remote(
        backend_config=backend_config
    )

    for i in range(request_num):
        await normal_backend.generate.remote(request_data=request_list[i])
    single_time = time.time() - start_time
    normal_backend.shutdown.remote()
    print(
        f"batch time: {batch_time*1000:.2f}ms, single time: {single_time*1000:.2f}ms"
    )

    del normal_backend

    assert batch_time < single_time


@pytest.mark.asyncio
async def test_resume_kv_cache(vllm_backend):
    new_request = {
        "model": "opt-125m",
        "prompt": long_prompt,
        "temperature": 0.3,
        "max_tokens": 100,
        "min_tokens": 100,
    }
    # zero request to start backend
    normal_request = await vllm_backend.generate.remote(
        request_data=request_data
    )

    # first generate without cache
    start = time.time()
    result = await vllm_backend.resume_kv_cache.remote([new_request])
    without_cache_latency = time.time() - start
    print(f"generate w/o cache latency: {without_cache_latency*1000:.2f}ms")

    # second generate with cache
    start = time.time()
    result = await vllm_backend.generate.remote(request_data=new_request)
    with_cache_latency = time.time() - start
    print(f"generate w/ cache latency: {with_cache_latency*1000:.2f}ms")

    assert with_cache_latency < without_cache_latency
