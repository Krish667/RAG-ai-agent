# Architecture

```text
User
  ↓
Streamlit UI
  ↓
PDF / DOCX / TXT
  ↓
Loaders → Text Splitter
  ↓
Gemini Embeddings
  ↓
Chroma Vector Database
  ↓
Similarity Search (RAG)
  ↓
Gemini LLM
  ↓
Answer + Source/Page
```
