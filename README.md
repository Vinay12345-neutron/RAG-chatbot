# 🤖 RAG Chatbot Project

## Overview
This project implements a simple yet functional **Retrieval Augmented Generation (RAG)** chatbot. The chatbot answers user queries based on the content of a provided research paper (PDF format). It combines **text retrieval** and **large language models (LLMs)** to generate accurate and contextually relevant responses.

The workflow consists of three stages:
1. **Ingestion**: Extract text from the PDF and split it into manageable chunks.
2. **Retrieval**: Use embeddings and vector similarity search to retrieve the most relevant chunks based on the user's query.
3. **Generation**: Pass the retrieved chunks to an LLM to generate a coherent response.

This project was created as part of an AI internship assignment to demonstrate practical knowledge of modern AI systems.

---

## Features
- **Knowledge Base**: Uses a single research paper (PDF) as its knowledge source.
- **Retrieval**: FAISS is used for efficient vector similarity search.
- **Generation**: OpenRouter API provides access to powerful LLMs like Mistral or Llama.
- **CLI Interface**: A simple command-line interface for interacting with the chatbot.

---

## Requirements
To run this project, you need:
- Python 3.8 or higher
- Access to an OpenRouter API key (or any other LLM API)
- A research paper in PDF format (e.g., `Orygin RAG AI Assignment.pdf`)

---

## Installation and Setup

### Step 1: Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```
---

### Step 2: Set Up a Virtual Environment
Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```
---

### Step 3: Install Dependencies
Install the required libraries using pip:
```bash
pip install -r requirements.txt
```
---


### Step 4: Add Your API Key
Create a .env file in the root directory and add your OpenRouter API key:
```bash
OPENROUTER_API_KEY=your_actual_api_key_here
```

---

### Step 5: Place the Research Paper
Place the research paper (PDF) in the root directory of the project and update the pdf_path variable in the code if necessary:
```bash
pdf_path = "Orygin RAG AI Assignment.pdf"
```

---

### How to Run the Chatbot
Run the chatbot script:
```bash
python rag_chatbot.py
```
Once the script is running, you’ll be prompted to ask questions. Type your query and press Enter to receive an answer. To exit the chatbot, type exit.

---

## Libraries and Frameworks Used
The following libraries and frameworks were used in this project:

- PyMuPDF (fitz) : For extracting text from the PDF.
- FAISS : For efficient vector similarity search.
- Sentence-Transformers : For generating text embeddings.
- LangChain : For splitting text into manageable chunks.
- OpenAI/OpenRouter : For interacting with large language models.
- python-dotenv : For managing environment variables.

## Workflow and Implementation Details
1. Ingestion
- The PDF is parsed using PyMuPDF , and the text is extracted page by page.
- The extracted text is split into smaller chunks using LangChain’s RecursiveCharacterTextSplitter . This ensures that each chunk is small enough to fit into the embedding model.
2. Retrieval
- Each text chunk is converted into a dense vector (embedding) using the sentence-transformers/all-MiniLM-L6-v2 model.
- These embeddings are stored in a FAISS index , which allows for fast similarity search.
- When a user asks a question, the query is also converted into an embedding, and the most relevant chunks are retrieved using cosine similarity.
3. Generation
- The retrieved chunks are passed to an LLM (via OpenRouter) along with the user’s query.
- The LLM generates a coherent and contextually relevant response based on the retrieved information

## Limitations
- The chatbot only answers questions based on the content of the provided research paper. It does not have access to external knowledge sources.
- The quality of responses depends on the accuracy of the retrieval process and the capabilities of the LLM.
- Large PDFs may require additional memory and processing time.

## Future Improvements
- Implement caching for embeddings to reduce computation time during subsequent runs.
- Add support for multiple documents or knowledge sources.
- Improve the retrieval process by experimenting with different embedding models or similarity metrics.
- Create a web-based GUI for better user interaction.

