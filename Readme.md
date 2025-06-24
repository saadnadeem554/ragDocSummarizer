# 🧠 RAG-Based Document Summarizer & Query Engine

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that reads documents, splits them into semantically meaningful chunks, indexes them using vector embeddings, and uses a powerful LLM to generate accurate, context-aware summaries or answers based on user queries.

---
## 🚀 Features

- 🔍 **Semantic Chunking** using LangChain's `RecursiveCharacterTextSplitter`
- 📄 **Multi-format Document Support**: `.pdf`, `.txt`, `.md`
- 🧠 **Sentence Embedding** using `SentenceTransformers`
- 📦 **FAISS Vector Store** for efficient semantic search
- 💬 **LLM Integration** with Groq's `llama3-70b-8192` for generation
- 🗂️ Optional **Persistent Vector Store** on disk
- 🧪 **CLI Support** for end-to-end processing and querying
- 🖥️ **Web Interface** with Streamlit for interactive document analysis

---

## 📦 Requirements

- Python 3.8+
- Install dependencies:

```bash
pip install -r requirements.txt
```

- Create a `.env` file with your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## 🛠 Usage

### 🔹 Basic Command

```bash
python document_summarizer.py path/to/document.pdf --query "Summarize the implementation details"
```

### 🔹 Advanced Options

```bash
python document_summarizer.py path/to/document.md \
  --query "Explain the architecture" \
  --chunk-size 2000 \
  --chunk-overlap 200 \
  --top-k 6 \
  --embedding-model "all-MiniLM-L6-v2" \
  --model "llama3-70b-8192" \
  --persist-dir "./vector_db"
```

| Argument              | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `file_path`           | Path to the input document (`.pdf`, `.txt`, `.md`)                          |
| `--query`             | The question or summary prompt to ask about the document                    |
| `--chunk-size`        | Size (in characters) of chunks to split document into                       |
| `--chunk-overlap`     | Overlap between chunks to preserve context                                  |
| `--top-k`             | Number of relevant chunks to retrieve for answering                         |
| `--model`             | LLM model name (default: `llama3-70b-8192`)                                 |
| `--embedding-model`   | Embedding model to use (default: `all-MiniLM-L6-v2`)                        |
| `--persist-dir`       | Directory to store/reuse FAISS vector index and chunks                      |
| `--reprocess`         | Force reprocessing of the document even if vector store exists              |

---

## 📚 How It Works

1. **Load** the document using format-specific LangChain loaders
2. **Split** into chunks based on semantic boundaries
3. **Embed** using a pretrained transformer model
4. **Store** in a FAISS vector index (optional persistence)
5. **Query** with a user prompt
6. **Retrieve** top-K relevant chunks via cosine similarity
7. **Generate** summary/answer using Groq LLM with retrieved context

---

## 📂 Example Output

```bash
Query: "Summarize the architecture of the pipeline"

✅ Response generated:

The system follows a modular Retrieval-Augmented Generation (RAG) architecture. It processes documents by loading and chunking them using semantic boundaries. Then it embeds the text into vector representations using SentenceTransformers and stores them in a FAISS index. A user query is embedded similarly, and the most relevant chunks are retrieved. Finally, the LLM is prompted with these chunks to generate a detailed summary or answer.
```

---

## 📁 Project Structure

```
.
├── document_summarizer.py   # Main script
├── requirements.txt         # Python dependencies
└── .env                     # Your Groq API Key
```

---

