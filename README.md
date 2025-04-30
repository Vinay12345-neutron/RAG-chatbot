# 🤖 RAG Chatbot Project

## Overview(Both Version 1 and Version 2 included)
This project implements a simple yet functional **Retrieval Augmented Generation (RAG)** chatbot. The chatbot answers user queries based on the content of a provided research paper (PDF format). It combines **text retrieval** and **large language models (LLMs)** to generate accurate and contextually relevant responses.

The workflow consists of three stages:
1. **Ingestion**: Extract text from the PDF and split it into manageable chunks.
2. **Retrieval**: Use embeddings and vector similarity search to retrieve the most relevant chunks based on the user's query.
3. **Generation**: Pass the retrieved chunks to an LLM to generate a coherent response.



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
- A research paper in PDF format (e.g., `Agentic_RAG.pdf`) has been uploaded.

---

## Installation and Setup

### Step 1: Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/Vinay12345-neutron/RAG-chatbot.git
cd RAG-chatbot
```
---

### Step 2: Set Up a Virtual Environment
Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:(bash)
source venv\Scripts\activate
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
pdf_path = "Agentic_RAG.pdf"
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

---

## Version 2 Updates
The second version of the RAG chatbot introduces several improvements to enhance performance, usability, and maintainability. Below is a summary of the key updates:

- ### Caching Embeddings :
Added support for caching embeddings to avoid recomputing them every time the script runs. This significantly reduces startup time, especially for large documents.
Embeddings are stored in a JSON file (embeddings_cache.json) and loaded dynamically when available.
- ### Improved Text Splitting :
Enhanced text splitting logic using langchain's RecursiveCharacterTextSplitter to handle complex document structures more effectively.
- ### Dynamic Chunk Size :
Introduced configurable chunk size and overlap parameters (CHUNK_SIZE and CHUNK_OVERLAP) for greater flexibility. These can be easily adjusted at the top of the script.
- ### Error Handling :
Added robust error handling for API calls, file operations, and other critical processes to ensure the chatbot runs smoothly even in edge cases.
- ### Answer Formatting :
Refined the prompt sent to the LLM to generate clearer, more concise, and well-structured answers.
- ### Logging :
Integrated logging to track the chatbot’s operations, debug issues, and provide insights into its workflow.
- ### User Feedback Loop :
Added a feedback mechanism to allow users to rate the quality of the generated answers. This helps identify areas for improvement and enhances user engagement.
Support for Multiple Documents (Future-Ready) :
The code is structured to support multiple PDFs as knowledge sources, making it easier to extend the chatbot’s capabilities in the future.
- ### Code Modularity :
Improved code modularity by organizing functions logically and ensuring each component (e.g., ingestion, retrieval, generation) is self-contained and reusable.
- ### How These Updates Improve the Chatbot
- Performance : Caching embeddings and optimizing text splitting reduce computation time and improve efficiency.
- Usability : The feedback loop and better answer formatting make the chatbot more interactive and user-friendly.
- Maintainability : Modular code and logging make it easier to debug, extend, and maintain the system.
- Scalability : The design supports future enhancements, such as adding multiple documents or deploying the chatbot to a web interface.



