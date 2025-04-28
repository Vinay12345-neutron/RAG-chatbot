import fitz  # PyMuPDF
from dotenv import load_dotenv
import os
from openai import OpenAI  # Explicitly import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize OpenRouter client
client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

def create_faiss_index(chunks):
    embeddings = SentenceTransformer('all-MiniLM-L6-v2').encode(chunks)
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

def retrieve_relevant_chunks(query, index, chunks, top_k=3):
    query_embedding = SentenceTransformer('all-MiniLM-L6-v2').encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

def generate_answer(query, relevant_chunks):
    context = "\n".join(relevant_chunks)
    prompt = f"""
    Answer the question based on the context below:
    Context: {context}
    Question: {query}
    Answer:
    """
    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

def main():
    pdf_path = "C:/Users/vinay/OneDrive/Desktop/VSCode/RAG/Agentic_RAG.pdf" # Replace with your PDF path
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text)
    index, embeddings = create_faiss_index(chunks)
    
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        query = input("\nAsk a question: ")
        if query.lower() == 'exit':
            break
        
        relevant_chunks = retrieve_relevant_chunks(query, index, chunks)
        answer = generate_answer(query, relevant_chunks)
        print(f"\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()