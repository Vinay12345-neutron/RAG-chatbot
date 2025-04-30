import fitz  # PyMuPDF
from dotenv import load_dotenv
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize OpenRouter client
client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
EMBEDDINGS_CACHE_FILE = "embeddings_cache.json"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        logging.info(f"Successfully extracted text from {pdf_path}")
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        raise

def split_text_into_chunks(text):
    """Splits text into manageable chunks using LangChain."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    logging.info(f"Split text into {len(chunks)} chunks")
    return chunks

def load_or_create_embeddings(chunks, cache_file=EMBEDDINGS_CACHE_FILE):
    """Loads embeddings from cache or generates and caches them."""
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached_data = json.load(f)
            logging.info("Loaded embeddings from cache")
            return cached_data["chunks"], np.array(cached_data["embeddings"])
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks).astype('float32')
    with open(cache_file, "w") as f:
        json.dump({"chunks": chunks, "embeddings": embeddings.tolist()}, f)
    logging.info("Generated and cached embeddings")
    return chunks, embeddings

def create_faiss_index(embeddings):
    """Creates a FAISS index for efficient similarity search."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    logging.info("Created FAISS index")
    return index

def retrieve_relevant_chunks(query, index, chunks, top_k=3):
    """Retrieves the most relevant chunks based on the query."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    logging.info(f"Retrieved {len(relevant_chunks)} relevant chunks")
    return relevant_chunks

def generate_answer(query, relevant_chunks):
    """Generates an answer using OpenRouter and the retrieved chunks."""
    context = "\n".join(relevant_chunks)
    prompt = f"""
    Based on the following context, provide a concise and accurate answer to the question:
    Context: {context}
    Question: {query}
    Answer:
    """
    try:
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        logging.info("Generated answer successfully")
        return answer
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        return "Sorry, I couldn't generate an answer due to an error."

def main():
    pdf_path = "C:/Users/vinay/OneDrive/Desktop/VSCode/RAG/Agentic_RAG.pdf"  # Replace with your PDF path
    try:
        # Step 1: Extract and process the PDF
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text_into_chunks(text)
        chunks, embeddings = load_or_create_embeddings(chunks)
        index = create_faiss_index(embeddings)

        # Step 2: Start the chatbot
        print("Chatbot is ready! Type 'exit' to quit.")
        while True:
            query = input("\nAsk a question: ").strip()
            if query.lower() == 'exit':
                break
            
            # Step 3: Retrieve and generate answer
            relevant_chunks = retrieve_relevant_chunks(query, index, chunks)
            answer = generate_answer(query, relevant_chunks)
            print(f"\nAnswer: {answer}\n")
            
            # Step 4: Collect user feedback
            feedback = input("Was this answer helpful? [y/n]: ").strip().lower()
            if feedback == "n":
                print("We'll try to improve!")
            elif feedback == "y":
                print("Thank you for your feedback!")
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()