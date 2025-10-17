# AI-Powered YouTube Tutor

This is a Streamlit app that lets you paste a YouTube video URL, fetches the transcript, chunks and embeds it with FAISS, and lets you ask questions about the video using an OpenAI model via LangChain.

## What it does
- Fetches the English transcript of a YouTube video (manual or auto subtitles)
- Splits the transcript into chunks and builds a FAISS vector index
- Uses OpenAI embeddings and an OpenAI LLM (via LangChain) to answer your questions
- Simple chat-style Q&A UI in the browser using Streamlit

## Requirements
- Python 3.10+ (3.11/3.12 also fine)
- An OpenAI API key with access to text-completion/chat models

## Setup (Windows PowerShell)
1. Create and activate a virtual environment (optional but recommended):
	- If you already have `vvenv`, activate it:
	  - PowerShell:
		 - `./vvenv/Scripts/Activate.ps1`
	- Or create a new one:
	  - `python -m venv .venv`
	  - `./.venv/Scripts/Activate.ps1`

2. Install dependencies:
	- `pip install -r requirements.txt`

3. Configure your API key:
	- Create a file named `.env` in the project root with:
	  - `OPENAI_API_KEY=your_key_here`

4. Run the app:
	- `streamlit run app.py`
	- Your browser will open at `http://localhost:8501`

## Usage
1. Paste a YouTube video URL (with available English transcript)
2. Click "Process Video" to fetch and index the transcript
3. Ask questions about the content in the input box

## Notes & Troubleshooting
- If you see "OPENAI_API_KEY not found", make sure `.env` exists and is in the project root (same folder as `app.py`). Restart the app after creating it.
- If the video has no English transcript or transcripts are disabled, the app will show an error.
- Corporate networks may block transcript retrieval; try a different network if you get region errors.
- FAISS is CPU-only by default (`faiss-cpu`).

## Security
- Do not commit your `.env` file. The `.gitignore` is configured to ignore it.