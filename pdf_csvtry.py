from google.colab import files
uploaded = files.upload()


import pdfplumber
import pandas as pd
import os

def extract_pdf_tables_and_text(path):
    c=1
    text_chunks, table_chunks = [], []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            # Extract tables
            print(c)
            tables = page.extract_tables()
            for tbl in tables:
                df = pd.DataFrame(tbl[1:], columns=tbl[0])
                table_chunks.append(df)
            # Extract text
            text = page.extract_text()
            if text:
                paragraphs = text.split('\n\n')
                text_chunks.extend(paragraphs)
            c+=1
    return text_chunks, table_chunks

def chunk_csv(df, chunk_size=10):
    return [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

text_chunks, table_chunks = [], []
metadata = []

for fname in uploaded.keys():
    print("start")
    ext = os.path.splitext(fname)[1].lower()
    print("skibidi")
    if ext == '.pdf':
        print("tt")
        texts, tables = extract_pdf_tables_and_text(fname)
        print(len(texts))
        for t in texts:
            print("Here")
            text_chunks.append(t)
            metadata.append({"type": "text", "source": fname})
        for tbl in tables:
            print("cooking")
            table_chunks.append(tbl)
            metadata.append({"type": "table", "source": fname})
    elif ext == '.xlsx':
       print("in xl")
       df = pd.read_excel(fname, engine='openpyxl')
       chunks = chunk_csv(df)
       for chunk in chunks:
           table_chunks.append(chunk)
           metadata.append({"type": "table", "source": fname})

def serialize_table(df):
    return "; ".join([f"{col}: {val}" for _, row in df.iterrows() for col, val in row.items()])
serialized_tables = [serialize_table(tbl) for tbl in table_chunks]
all_chunks = text_chunks + serialized_tables
all_metadata = metadata[:len(text_chunks)] + metadata[len(text_chunks):]

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
embeddings = model.encode(all_chunks, show_progress_bar=True)

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

client = QdrantClient(":memory:")
client.recreate_collection(
    collection_name="private_docs",
    vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE),
)

points = [
    PointStruct(
        id=i,
        vector=embeddings[i].tolist(),
        payload={"text": all_chunks[i], **all_metadata[i]}
    )
    for i in range(len(all_chunks))
]
client.upsert(collection_name="private_docs", points=points)

def search_query(query, top_k=5):
    query_vec = model.encode([query])[0].tolist()
    results = client.search(collection_name="private_docs", query_vector=query_vec, limit=top_k)
    return results

query = input("Enter query")
results = search_query(query)

for i, res in enumerate(results):
    print(f"\n--- Result #{i+1} ---")
    print(f"Score: {res.score:.3f}")
    print(f"Type: {res.payload['type']}")
    print(f"Source: {res.payload['source']}")
    print("Content:")
    print(res.payload['text'][:1000])
