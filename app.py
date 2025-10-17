import os
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from pytube import YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled, 
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Check for API key
if not api_key:
    st.error("OPENAI_API_KEY not found in .env file.")
    st.stop()

st.title("🎓 AI-Powered YouTube Tutor")
st.write("Ask questions from YouTube lecture transcripts using LangChain + OpenAI.")

video_url = st.text_input("📺 Enter YouTube Video URL")

# ------------------ Transcript Fetching Function ------------------
def get_youtube_transcript(url):
    try:
        video_id = YouTube(url).video_id
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Find the English transcript (auto or manual)
        transcript = transcript_list.find_transcript(["en"])
        transcript_data = transcript.fetch()

        # Ensure it's in expected list of dicts format
        if isinstance(transcript_data, list) and all("text" in item for item in transcript_data):
            return " ".join([item["text"] for item in transcript_data])
        else:
            raise ValueError("Transcript data is not in expected format (list of dicts).")

    except TranscriptsDisabled:
        st.error("❌ Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("❌ No transcript found for this video.")
    except VideoUnavailable:
        st.error("❌ This video is unavailable.")
    except CouldNotRetrieveTranscript:
        st.error("❌ Could not retrieve transcript. May not be available in your region.")
    except Exception as e:
        st.error(f"❌ Unexpected error getting transcript: {e}")
    return ""

# ------------------ Save Transcript to File ------------------
def save_transcript_to_file(text, filename="transcript.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

# ------------------ Main App Flow ------------------
if st.button("📄 Process Video"):
    if video_url:
        transcript_text = get_youtube_transcript(video_url)
        if transcript_text:
            save_transcript_to_file(transcript_text)

            # Load and split documents
            loader = TextLoader("transcript.txt", encoding="utf-8")
            documents = loader.load()
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(documents)

            # Create vectorstore and retrieval chain
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever()

            llm = OpenAI(openai_api_key=api_key)
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

            st.session_state.qa_chain = qa_chain
            st.success("✅ Transcript processed! Ask your questions below.")
    else:
        st.warning("⚠️ Please enter a valid YouTube URL.")

# ------------------ Q&A Section ------------------
if "qa_chain" in st.session_state:
    user_question = st.text_input("❓ Ask a question about the video")
    if user_question:
        try:
            answer = st.session_state.qa_chain.run(user_question)
            st.write("**💡 Answer:**", answer)
        except Exception as e:
            st.error(f"❌ Error answering question: {e}")
