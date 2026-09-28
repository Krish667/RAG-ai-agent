# API / Application Interface

This project is a Streamlit application rather than a separate REST backend.

| Function | Purpose |
|---|---|
| `process_document(file_path)` | Loads PDF/DOCX/TXT, splits text and stores embeddings in Chroma. |
| `answer_question(query)` | Retrieves relevant chunks and sends context to the Google Gemini LLM. |
| Streamlit uploader | Accepts multiple documents. |
| Admin Dashboard | Deletes documents, shows statistics and response history. |
