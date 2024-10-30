# app/main.py
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.rag import RAGSystem
from src.core.llm import get_llm_and_embeddings
from src.utils.file_handler import process_file

def setup_page():
   st.set_page_config(
       page_title="RAG Chatbot",
       layout="centered",
       initial_sidebar_state="expanded"
   )
   st.markdown("""
       <style>
       .stApp { max-width: 1200px; margin: 0 auto; }
       .chat-message {
           padding: 1.5rem; 
           border-radius: 0.5rem;
           margin-bottom: 1rem;
           box-shadow: 0 2px 4px rgba(0,0,0,0.1);
       }
       .user-message { background: #f0f7ff; }
       .assistant-message { background: #f8f9fa; }
       .sidebar .stButton>button {
           width: 100%;
           margin-top: 1rem;
       }
       </style>
       """, unsafe_allow_html=True)

def initialize_session_state():
   if "messages" not in st.session_state:
       st.session_state.messages = []
   if "chain" not in st.session_state:
       st.session_state.chain = None
   if "api_choice" not in st.session_state:
       st.session_state.api_choice = None

def handle_chat(prompt, chain):
   try:
       chat_history = [
           (msg["content"], next_msg["content"])
           for i, msg in enumerate(st.session_state.messages[:-1])
           if msg["role"] == "user" 
           and (next_msg := st.session_state.messages[i + 1])["role"] == "assistant"
       ]
       
       st.session_state.messages.append({"role": "user", "content": prompt})
       
       response = chain({"question": prompt, "chat_history": chat_history})
       
       st.session_state.messages.append({
           "role": "assistant",
           "content": response["answer"],
           "sources": response.get("source_documents", [])
       })
       
   except Exception as e:
       st.error(f"Error processing request: {str(e)}")
       return
       
   st.experimental_rerun()

def display_chat():
   for message in st.session_state.messages:
       with st.chat_message(message["role"]):
           st.markdown(message["content"])
           if message["role"] == "assistant" and "sources" in message:
               with st.expander("View Sources"):
                   for idx, source in enumerate(message["sources"], 1):
                       st.markdown(f"**Source {idx}:**")
                       st.code(source, language="text")

def main():
   setup_page()
   initialize_session_state()

   st.title("Chat Assistant")

   with st.sidebar:
       st.header("Configuration")
       api_choice = st.selectbox("Select API", ["OpenAI", "Gemini"])
       api_key = st.text_input("Enter API Key", type="password")

       if api_choice != st.session_state.api_choice:
           st.session_state.chain = None
           st.session_state.api_choice = api_choice

       uploaded_files = st.file_uploader(
           "Upload Documents",
           accept_multiple_files=True, 
           type=["txt", "pdf"],
           help="Upload PDF or TXT files to use as knowledge base"
       )

       if st.button("Clear Chat"):
           st.session_state.messages = []
           st.session_state.chain = None

       if uploaded_files and api_key:
           if st.button("Process Documents"):
               try:
                   with st.spinner("Processing documents..."):
                       llm, embeddings = get_llm_and_embeddings(api_choice, api_key)
                       documents = []
                       for file in uploaded_files:
                           documents.extend(process_file(file))

                       rag_system = RAGSystem(llm, embeddings)
                       vectorstore = rag_system.create_vectorstore(documents)
                       st.session_state.chain = rag_system.get_chain(vectorstore)
                       st.success("Documents processed successfully!")
               except Exception as e:
                   st.error(f"Error processing documents: {str(e)}")

   if not st.session_state.chain:
       st.info("Please upload documents and enter API key to start chatting.")
       return

   display_chat()

   if prompt := st.chat_input("Ask a question about your documents..."):
       with st.spinner("Thinking..."):
           handle_chat(prompt, st.session_state.chain)

if __name__ == "__main__":
   main()