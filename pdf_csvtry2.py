from google.colab import files
uploaded = files.upload()

from unstructured.partition.pdf import partition_pdf
import pdfplumber
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def extract_with_unstructured(path):
    
    # Extract all layout elements (headers, paragraphs, etc.)
    elements = partition_pdf(filename=path)
    # elements is a list of objects with .type and .text attributes
    return elements

def extract_pdf_tables(path):
    tables = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_tables = page.extract_tables()
            for tbl in page_tables:
                df = pd.DataFrame(tbl[1:], columns=tbl[0])
                tables.append((page_num+1, df))  # store page number with table
    return tables

def chunk_text_with_headers(elements):
    chunks = []
    current_header = None
    current_text = []
    for el in elements:
        el_type = type(el).__name__
        if el_type in ['Heading1', 'Heading2', 'Heading3']:
            # save previous chunk
            if current_text:
                chunks.append({'header': current_header, 'text': "\n".join(current_text)})
                current_text = []
            current_header = el.text.strip()
        elif el_type == 'Paragraph':
            current_text.append(el.text.strip())
    # Add last chunk
    if current_text:
        chunks.append({'header': current_header, 'text': "\n".join(current_text)})
    return chunks

def extract_keywords(text, top_n=5):
    # Simple keyword extraction using TF-IDF (or any other)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
    X = vectorizer.fit_transform([text])
    features = vectorizer.get_feature_names_out()
    scores = X.toarray()[0]
    # Pick top_n words
    keywords = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)[:top_n]
    return [kw for kw, score in keywords]

def serialize_table(df):
    return "; ".join([f"{col}: {val}" for _, row in df.iterrows() for col, val in row.items()])

# Main processing function
def process_pdf(path):
    print("yo")
    elements = extract_with_unstructured(path)
    tables = extract_pdf_tables(path)
    text_chunks = chunk_text_with_headers(elements)
    print("ongoing")
    all_chunks = []
    metadata = []

    for chunk in text_chunks:
        header = chunk['header'] or ""
        text = chunk['text']
        keywords = extract_keywords(text)
        # Find tables on same page or link by heuristics if possible
        # (For simplicity, we skip table linking here)
        combined_text = (header + "\n" + text).strip()
        all_chunks.append(combined_text)
        metadata.append({
            "type": "text",
            "source": os.path.basename(path),
            "header": header,
            "keywords": keywords,
        })

    # Add tables as separate chunks with metadata
    for page_num, tbl in tables:
        serialized = serialize_table(tbl)
        all_chunks.append(serialized)
        metadata.append({
            "type": "table",
            "source": os.path.basename(path),
            "page": page_num,
            "header": None,
            "keywords": [],
        })
    return all_chunks, metadata

all_chunks, all_metadata = [], []

for fname in uploaded.keys():
    ext = os.path.splitext(fname)[1].lower()
    if ext == '.pdf':
        all_chunks, all_metadata = process_pdf(fname)
    elif ext in ['.csv', '.xlsx']:
        df = pd.read_csv(fname) if ext == '.csv' else pd.read_excel(fname, engine='openpyxl')
        table_chunks = chunk_csv(df)
        for tbl in table_chunks:
            serialized = "; ".join([f"{col}: {val}" for _, row in tbl.iterrows() for col, val in row.items()])
            all_chunks.append(serialized)
            all_metadata.append({"type": "table", "source": fname})

def chunk_csv(df, chunk_size=10):
    return [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]




from sentence_transformers import SentenceTransformer, CrossEncoder
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

def search_query(query, top_k=10):
    query_vec = model.encode([query])[0].tolist()
    results = client.search(collection_name="private_docs", query_vector=query_vec, limit=top_k)
    return results


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_results(query, results):
    pairs = [(query, res.payload['text']) for res in results]
    scores = reranker.predict(pairs)
    return [res for _, res in sorted(zip(scores, results), key=lambda x: x[0], reverse=True)]

query = input("Enter query")
initial_results = search_query(query)
ranked_results = rerank_results(query, initial_results)

for i, res in enumerate(ranked_results):
    print(f"\n--- Result #{i+1} ---")
    print(f"Score (vector match): {res.score:.3f}")
    print(f"Type: {res.payload.get('type')} | Taxa: {res.payload.get('taxa')} | Topic: {res.payload.get('topic')}")
    print(f"Source: {res.payload.get('source')} | Section: {res.payload.get('section')}")
    print("Content:")
    print(res.payload['text'][:1000])
