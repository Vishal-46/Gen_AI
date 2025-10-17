# AI-Powered YouTube Tutor

This is a Streamlit app that lets you paste a YouTube video URL, fetches the transcript, chunks and embeds it with FAISS, and lets you ask questions about the video using an OpenAI model via LangChain.

## What it does
- Fetches the English transcript of a YouTube video (manual or auto subtitles)
- Splits the transcript into chunks and builds a FAISS vector index
- Uses OpenAI embeddings and an OpenAI LLM (via LangChain) to answer your questions
- Simple chat-style Q&A UI in the browser using Streamlit
