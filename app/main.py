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
        page_title="DocuChat",
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
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .user-message { background: #e6f7ff; border-left: 5px solid #1890ff; }
        .assistant-message { background: #f9f9f9; border-left: 5px solid #52c41a; }
        .sidebar .stButton>button {
            width: 100%;
            margin-top: 1rem;
            border-radius: 0.5rem;
        }
        .stTextInput>div>div>input {
            border-radius: 0.5rem;
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
           "content": response.get("answer", "Sorry, I couldn't find an answer."),
           "sources": response.get("source_documents", [])
       })
       
   except Exception as e:
       st.error(f"Error processing request: {str(e)}")
       return
       
   st.rerun()

def display_chat():
   for message in st.session_state.messages:
       with st.chat_message(message["role"]):
           st.markdown(message["content"])
           if message["role"] == "assistant" and message.get("sources"):
               with st.expander("View Sources"):
                   for idx, source in enumerate(message["sources"], 1):
                       source_name = source.metadata.get('source', 'Unknown Source')
                       st.markdown(f"**Source {idx}: {source_name}**")
                       st.code(source.page_content, language="text")

def main():
   setup_page()
   initialize_session_state()

   st.title("📄 DocuChat: Your Personal Document Assistant")
   st.write("""
       Welcome to **DocuChat**, your intelligent assistant for understanding and interacting with your documents.
       Upload your files, and I'll help you find the information you need.
   """)

   with st.sidebar:
       st.header("⚙️ Configuration")
       st.markdown("---")
       st.subheader("How It Works")
       st.info("""
           1.  **Select your API**: Choose between OpenAI or Gemini.
           2.  **Enter your API key**: Securely enter your key to power the chat.
           3.  **Upload your documents**: PDF and TXT files are supported.
           4.  **Process and chat**: Click 'Process' and start asking questions!
       """)
       st.markdown("---")
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