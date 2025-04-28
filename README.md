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