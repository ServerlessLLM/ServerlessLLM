import torch
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
client = OpenAI(
    base_url="http://127.0.0.1:8343/v1",
)

chat_model = "Qwen/Qwen2.5-7B-Instruct"
embedding_model = "intfloat/e5-mistral-7b-instruct"
relevance_expert = "sentence-transformers/all-MiniLM-L6-v2"


## Define functions
def encode(texts):
    if isinstance(texts, str):
        batch = [texts]
    else:
        batch = texts
    res = client.embeddings.create(input=batch, model=embedding_model)
    return [i.embedding for i in res.data]


def select_rel(passages, query, r):
    def compute_relevance_scores(query_encoding, document_encodings, r):
        scores = torch.matmul(
            query_encoding.unsqueeze(0), document_encodings.transpose(1, 2)
        )
        max_scores_per_query_term = scores.max(dim=2).values
        total_scores = max_scores_per_query_term.sum(dim=1)
        sorted_indices = total_scores.argsort(descending=True)
        return sorted_indices[:r]

    query_encoding = (
        client.embeddings.create(input=[query], model=relevance_expert)
        .data[0]
        .embedding
    )
    query_encoding = torch.tensor([query_encoding])

    passage_encoding_res = client.embeddings.create(
        input=[passages], model=relevance_expert
    )
    passage_encodings = torch.tensor(
        [i.embedding for i in passage_encoding_res.data]
    )
    relevant_indices = compute_relevance_scores(
        query_encoding, passage_encodings, r
    )
    relevant_passages = [passages[i] for i in relevant_indices]
    return relevant_passages


def chat(relevant_passages, query):
    RAG_prompt = f"""
    Using the following information ranked from most to least relevant (top to bottom):
    {relevant_passages}.

    Answer the following question, if possible: {query}."""

    completion = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": f"{RAG_prompt}"}],
        max_tokens=100,
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
relevant_passages = select_rel(passages, query, 2)

# Generate the final answer.
answer = chat(relevant_passages, query)

print(answer)
