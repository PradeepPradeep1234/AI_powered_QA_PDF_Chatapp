import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
from huggingface_hub import login
login(token=st.secrets["HF_TOKEN"])

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Load embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Helper: Read PDF and split into chunks
def read_pdf_chunks(pdf_file, chunk_size=300):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    sentences = text.split(". ")
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current.split()) + len(sentence.split()) <= chunk_size:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks

# Helper: Embed chunks using sentence-transformers
def embed_chunks(chunks):
    return embed_model.encode(chunks)

# Helper: Search top relevant chunks using FAISS
def search_similar_chunks(question, chunks, chunk_vectors, k=3):
    index = faiss.IndexFlatL2(chunk_vectors.shape[1])
    index.add(chunk_vectors)
    q_embedding = embed_model.encode([question])
    _, indices = index.search(q_embedding, k)
    return [chunks[i] for i in indices[0]]

# Helper: Query Groq + Mixtral
def query_groq(context, question):
    prompt = f"""
Use only the information below to answer the question. If not available, say "Information not found in the PDF".

Context:
{context}

Question: {question}
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant who answers based only on the given context."},
            {"role": "user", "content": prompt}
        ],
        "model": "llama3-8b-8192",
        "temperature": 0.3,
        "max_tokens": 300
    }

    res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    if res.status_code == 200:
        return res.json()["choices"][0]["message"]["content"]
    else:
        return f"Error {res.status_code}: {res.text}"



# Streamlit UI
st.set_page_config(page_title="Ask Questions from PDF", layout="centered")
st.title("📄 Ask Questions from PDF (Groq + Mixtral)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    st.success("PDF uploaded successfully!")

    question = st.text_input("Ask a question based on this PDF")

    if question:
        with st.spinner("Reading and processing PDF..."):
            chunks = read_pdf_chunks(uploaded_file)
            chunk_vectors = embed_chunks(chunks)
            top_chunks = search_similar_chunks(question, chunks, np.array(chunk_vectors))

        with st.spinner("Getting answer from Groq..."):
            context = "\n".join(top_chunks)
            answer = query_groq(context, question)
            st.subheader("📢 Answer:")
            st.write(answer)
