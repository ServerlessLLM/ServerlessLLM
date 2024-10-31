from openai import OpenAI
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, HnswConfigDiff
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json

api_key = "sk-yb6QzaXktLjJblSPE411967f5a5d4e0d83598a85F9C3Be03"
client_emb = OpenAI(
    base_url="http://localhost:8000/v1",
)

client_chat = OpenAI(
    base_url="http://localhost:8001/v1",
)

client_rerank = OpenAI(
    base_url="http://localhost:8002/v1",
)

chat_model = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8"
embedding_model = "intfloat/e5-mistral-7b-instruct"
relevance_expert = "Qwen/Qwen2-7B-Instruct-GPTQ-Int8"

wiki_passages = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
docs = wiki_passages["passages"]["passage"]
query = "How many long was Lincoln's formal education?"
k = 5

client = QdrantClient(path="/home/ubuntu/qdrant-store")
collection_name = "wiki"

if not client.collection_exists(collection_name):
    client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=4096, distance=Distance.EUCLID),
    hnsw_config=HnswConfigDiff(
        m=16, 
        ef_construct=100,
        full_scan_threshold=10000
        )
    )

    embedding_lst = []
    
    if os.path.exists("embedding_lst.pkl"):
        embedding_lst = pickle.load(open("embedding_lst.pkl", "rb"))
    else:
        batches = [docs[i:i+32] for i in range(0, len(docs), 32)]
        for batch in tqdm(batches, desc="Embedding"):
            res = client_emb.embeddings.create(
                input=batch,
                model=embedding_model
            )
            embedding_lst.extend([i.embedding for i in res.data])

        pickle.dump(embedding_lst, open("embedding_lst.pkl", "wb"))
        
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=i,
                vector=embedding
            )
            for i, embedding in enumerate(embedding_lst)
        ]
    )

query_res = client_emb.embeddings.create(
    input=[query],
    model=embedding_model
)

query_embedding = query_res.data[0].embedding

similar_doc_ids = client.search(
    collection_name=collection_name,
    query_vector=query_embedding,
    limit=k
)

passages = [docs[i.id] for i in similar_doc_ids]

passage_order_str = '\n'.join([f"[{i}] {passage}" for i, passage in enumerate(passages)])

relevance_prompts = [{"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": f'''
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
'''}
]

rerank_completion = client_rerank.chat.completions.create(
  model=relevance_expert,
  messages=relevance_prompts
)

rerank_res = rerank_completion.choices[0].message.content

rerank_res = json.loads(rerank_res)

reranked_passages = [passages[i] for i in rerank_res]
reranked_texts = '\n'.join(reranked_passages)

RAG_prompt = f'''
Using the following information ranked from most to least relevant (top to bottom): 
{reranked_texts}.

Answer the following question, if possible: {query}.'''

completion = client_chat.chat.completions.create(
  model=chat_model,
  messages=[
    {"role": "user", "content": f"{RAG_prompt}"}
  ]
)

print(completion.choices[0].message.content)