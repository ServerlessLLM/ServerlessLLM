import json

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


## Define functions
def encode(texts):
    if isinstance(texts, str):
        batch = [texts]
    res = client_emb.embeddings.create(input=batch, model=embedding_model)
    return [i.embedding for i in res.data]


def select_rel(passages, query, k, r=2):
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
    Which {r} passages are the most relevant to answering the following question?
    {query}

    **Output format**:
    Return only a list of the {r} most relevant passage IDs as integers from most relevant to least relevant (e.g., `[ID1, ID2]`).
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
    return relevant_passages


def chat(relevant_passages, query):
    RAG_prompt = f"""
    Using the following information ranked from most to least relevant (top to bottom):
    {relevant_passages}.

    Answer the following question, if possible: {query}."""

    completion = client_chat.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": f"{RAG_prompt}"}],
    )
    return completion.choices[0].message.content


## Load dataset
wiki_passages = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
docs = wiki_passages["passages"]["passage"]
query = "How many long was Lincoln's formal education?"
k = 5

## Build VectorDB store with Qdrant
qdrant_cli = QdrantClient(path="qdrant-store")
collection_name = "wiki"

if not qdrant_cli.collection_exists(collection_name):
    qdrant_cli.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=4096, distance=Distance.EUCLID),
        hnsw_config=HnswConfigDiff(
            m=16, ef_construct=100, full_scan_threshold=10000
        ),
    )

    ## Embed the documents and insert them into the Qdrant collection
    embedding_lst = []
    batch_size = 32
    batches = [
        docs[i : i + batch_size] for i in range(0, len(docs), batch_size)
    ]
    for batch in tqdm(batches, desc="Embedding"):
        res = encode(batch)
        embedding_lst.extend(res)

    qdrant_cli.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(id=i, vector=embedding)
            for i, embedding in enumerate(embedding_lst)
        ],
    )

## Get query embedding
query_res = encode(query)
query_embedding = query_res[0]

## Search for similar documents
similar_doc_ids = qdrant_cli.search(
    collection_name=collection_name, query_vector=query_embedding, limit=k
)
passages = [docs[i.id] for i in similar_doc_ids]

## Select the most relevant passages
relevant_passages = select_rel(passages, query, k)

## Generate the final answer.
answer = chat(relevant_passages, query)

print(answer)
