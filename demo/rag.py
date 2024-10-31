import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

## Create OpenAI clients
client_emb = OpenAI(
    base_url="http://localhost:8000/v1",
)

client_chat = OpenAI(
    base_url="http://localhost:8001/v1",
)

client_relevance = OpenAI(
    base_url="http://localhost:8002/v1",
)

chat_model = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8"
embedding_model = "intfloat/e5-mistral-7b-instruct"
relevance_expert = "Qwen/Qwen2-7B-Instruct-GPTQ-Int8"

## Load dataset
wiki_passages = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
docs = wiki_passages["passages"]["passage"]
query = "How many long was Lincoln's formal education?"
k = 5

## Build VectorDB store with Qdrant
client = QdrantClient(path="qdrant-store")
collection_name = "wiki"

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=4096, distance=Distance.EUCLID),
        hnsw_config=HnswConfigDiff(
            m=16, ef_construct=100, full_scan_threshold=10000
        ),
    )

    embedding_lst = []
    batch_size = 32

    if os.path.exists("embedding_lst.pkl"):
        embedding_lst = pickle.load(open("embedding_lst.pkl", "rb"))
    else:
        batches = [
            docs[i : i + batch_size] for i in range(0, len(docs), batch_size)
        ]
        for batch in tqdm(batches, desc="Embedding"):
            res = client_emb.embeddings.create(
                input=batch, model=embedding_model
            )
            embedding_lst.extend([i.embedding for i in res.data])

        pickle.dump(embedding_lst, open("embedding_lst.pkl", "wb"))

    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(id=i, vector=embedding)
            for i, embedding in enumerate(embedding_lst)
        ],
    )

## Get query embedding
query_res = client_emb.embeddings.create(input=[query], model=embedding_model)

query_embedding = query_res.data[0].embedding

## Search for similar documents
similar_doc_ids = client.search(
    collection_name=collection_name, query_vector=query_embedding, limit=k
)

passages = [docs[i.id] for i in similar_doc_ids]

## Select the most relevant passages
passage_order_str = "\n".join(
    [f"[{i}] {passage}" for i, passage in enumerate(passages)]
)
relevance_prompts = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": f"""
**Important**:
The output should be only a list of two integer IDs.

I will provide you with {k} passages, each labeled with a unique integer ID (e.g., `[ID] Passage`).

**Passages**:
{passage_order_str}

**Question**:
Which two passages are the most relevant to answering the following question?
{query}

**Output format**:
Return only a list of the two most relevant passage IDs as integers (e.g., `[ID1, ID2]`).
""",
    },
]

relevance_check_completion = client_relevance.chat.completions.create(
    model=relevance_expert, messages=relevance_prompts
)

relevance_check_res = relevance_check_completion.choices[0].message.content
relevance_check_res = json.loads(relevance_check_res)

relevant_passages = [passages[i] for i in relevance_check_res]
relevant_passages = "\n".join(relevant_passages)

## Generate the final answer.
RAG_prompt = f"""
Using the following information ranked from most to least relevant (top to bottom):
{relevant_passages}.

Answer the following question, if possible: {query}."""

completion = client_chat.chat.completions.create(
    model=chat_model, messages=[{"role": "user", "content": f"{RAG_prompt}"}]
)

print(completion.choices[0].message.content)
