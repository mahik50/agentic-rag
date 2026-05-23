# 🧠 PageMind - Agentic RAG Webpage Q&A Chatbot
> Ask anything about any webpage. PageMind scrapes, indexes, and lets you have a full conversation with any URL.
## 🚀 Overview
**PageMind** is an AI-powered, agentic Retrieval-Augmented Generation (RAG) chatbot that allows users to input any webpage URL and ask natural language questions about its content. It intelligently routes queries through different retrieval strategies, self-corrects when context is insufficient, and maintains full multi-turn conversation history.
## ✨ Features
- 🔗 **URL Ingestion** - Recursively scrapes and indexes any webpage using LangChain's `RecursiveUrlLoader`
- 🧩 **Smart Chunking** - Splits content into overlapping chunks for precise semantic retrieval
- 🗄️ **Persistent Vector Store** - Stores embeddings in ChromaDB with caching to avoid re-indexing
- 🤖 **Agentic Query Routing** - Automatically classifies queries into three strategies:
 - `summary` - High-level overview of the entire page
 - `section` - Targeted semantic search for specific facts
 - `exhaustive` - Full-page scan for complete extraction
- 🔁 **Self-Correction Loop** - Evaluates retrieved context for sufficiency; rewrites and retries if insufficient
- 💬 **Multi-turn Conversation** - Maintains full chat history for follow-up questions
- 🖥️ **Streamlit UI** - Clean, interactive frontend for real-time Q&A
 - -
## 🏗️ Architecture
```
User Query
 │
 ▼
[Router Node] ──────────────────────────────────────┐
 │                                                  │
 ├──► [Summary Node] (full-page summary)            │
 ├──► [Section Node] (top-k semantic search)        │
 └──► [Exhaustive Node] (scan all chunks)           │
                                                    │
 ◄──────────────────────────────────────────────────┘
 │
 ▼
[Self-Correct Node]
 │
 ├──► sufficient ──► [Generate Node] ──► Answer
 └──► insufficient ──► [Refine Node] ──► [Router Node] (retry)
```

## ▶️ Usage
```bash
# Run the Streamlit app
streamlit run frontend.py
```
1. Enter any URL in the input box
2. Wait for the page to be scraped and indexed
3. Ask questions in the chat interface!
on**: The URL is scraped recursively, cleaned with BeautifulSoup, split into chunks, embedded, and stored in ChromaDB.
2. **Routing**: Each user query is classified by an LLM into `summary`, `section`, or `exhaustive` retrieval mode.
3. **Retrieval**: The appropriate retriever fetches context from the vector store.
4. **Self-Correction**: An LLM evaluates whether the retrieved context is sufficient. If not, the query is rewritten and retried.
5. **Generation**: The final answer is generated using the full conversation history and retrieved context.
