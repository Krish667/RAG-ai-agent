# AI Document Intelligence Assistant (RAG)

A simple Streamlit RAG application based on the original assignment code.

## Features
- PDF, DOCX and TXT upload
- Multiple document upload
- Text extraction, chunking and Gemini embeddings
- Persistent Chroma vector database
- RAG question answering with Google Gemini
- Source document and PDF page number
- Chat history
- Admin dashboard: upload, delete, statistics and response history

## Run
1. Create a virtual environment.
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your Google Gemini API key.
4. Run: `streamlit run app.py`

## Deployment
Deploy `app.py` on Streamlit Community Cloud, add `GOOGLE_API_KEY` in Secrets, then copy the generated app URL into your submission.
