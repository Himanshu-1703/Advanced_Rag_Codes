# 📚 Advanced RAG Codes

A comprehensive collection of code implementations covering the full spectrum of **Retrieval-Augmented Generation (RAG)** techniques — from foundational building blocks to advanced, production-ready patterns.

---

## 🗂️ Repository Structure

| Folder | Topic |
|--------|-------|
| `01_document_loaders` | Loading documents from various sources (PDF, Web, JSON etc.) |
| `02_text_splitters` | Chunking strategies to split documents into meaningful pieces |
| `03_embeddings` | Generating vector embeddings using different embedding models |
| `04_vector_stores` | Storing and querying embeddings using vector databases |
| `05_retrievers` | Core retrieval strategies to fetch relevant document chunks |
| `06_advanced_retrievers` | Advanced retrieval techniques like parent-document, multi-query, self-query and contextual compression retrievers |
| `07_rerankers` | Re-ranking retrieved results to improve relevance using cross-encoders models |
| `08_rag_fusion` | Combining results from multiple retrievers using fusion strategies like RRF |
| `09_rag_hyde` | HyDE (Hypothetical Document Embeddings) — generating hypothetical answers to improve retrieval |
| `10_agentic_rag` | Agentic RAG patterns where LLMs autonomously decide when and how to retrieve |
| `11_graph_rag` | Graph-based RAG using knowledge graphs for structured information retrieval |
| `12_rag_multimodal` | Multimodal RAG handling text, images and tables in a single document |
| `13_rag_evaluation` | Evaluation frameworks (RAGAS) and metrics to measure RAG pipeline performance |
| `14_rag_papeer_project` | End-to-end RAG project built around research papers assistance and fact/claim verification |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- LangChain, LangGraph
- An API key for your preferred LLM provider (OpenAI, Anthropic, etc.)

### Installation

Clone the repository:

```bash
git clone https://github.com/Himanshu-1703/Advanced_Rag_Codes.git
cd Advanced_Rag_Codes
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```
OR
```bash
uv sync
```

Set up your environment variables:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

---

## 📖 How to Use

Each folder is self-contained and corresponds to a specific RAG topic. Navigate into any folder and run the Jupyter notebooks in order:

```bash
cd 05_retrievers
jupyter notebook
```

The folders are numbered to reflect a **recommended learning path** — start from `01` and work your way through to build a solid understanding of each concept before moving to the next.

---

## 🧠 Topics Covered

- **Document Ingestion** — Loading and preprocessing data from multiple sources
- **Chunking Strategies** — Fixed-size, recursive, semantic, document structure based and LLM based splitting
- **Embedding Models** — OpenAI, Ollama and output dimensionality tradeoff
- **Vector Databases** — Embeddings Persistance and Indexing techniques
- **Retrieval Techniques** — BM25, dense retrieval, hybrid search
- **Advanced Retrievers** — Parent-document, multi-query, self-query, ensemble retrievers
- **Re-ranking** — Cross-encoder rerankers to boost retrieval precision
- **RAG Fusion** — Merging multiple retrieval results with Reciprocal Rank Fusion (RRF)
- **HyDE** — Improving retrieval by generating hypothetical documents
- **Agentic RAG** — Tool-calling agents that decide retrieval strategy dynamically
- **Graph RAG** — Leveraging knowledge graphs alongside vector retrieval
- **Multimodal RAG** — Processing and retrieving across text and image modalities
- **RAG Evaluation** — RAGAS and other frameworks for measuring faithfulness, relevance, and correctness

---

## 🤝 Contributing

Contributions are welcome! If you find a bug or want to add improvements:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to your branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## ⭐ Support

If you find this repository helpful, please consider giving it a **star** ⭐ — it helps others discover it!

---

## 📄 License

This project is licensed under the MIT License.
