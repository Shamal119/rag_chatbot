# 📄 DocuChat: Your Personal Document Assistant

Welcome to **DocuChat**, an intelligent assistant that helps you understand and interact with your documents. Upload your files (PDF or TXT), and DocuChat will answer your questions based on the content of those documents.

## 🚀 How It Works

1.  **Select your AI Provider**: Choose between OpenAI or Gemini to power the chat.
2.  **Enter your API Key**: Securely provide your API key.
3.  **Upload Your Documents**: Upload one or more PDF or TXT files.
4.  **Process and Chat**: Click "Process" to create a knowledge base, and then start asking questions!

## 🛠️ Technologies Used

This project is built with a few key technologies to make the magic happen:

*   **Python**: The core programming language that powers the entire application.
*   **Streamlit**: Used to create the interactive and user-friendly web interface you see. It's a simple way to build and share data apps.
*   **LangChain**: The framework that connects everything. It helps us chain together the language model, our documents, and the logic to answer your questions.
*   **OpenAI & Gemini**: You have the choice of using either of these powerful Large Language Models (LLMs) as the "brain" of the chatbot.
*   **FAISS (Facebook AI Similarity Search)**: A library for efficient similarity search. We use it to create a vector store of your documents, which allows us to quickly find the most relevant information to answer your questions.
*   **PyPDF2**: A Python library used to read and extract text from your PDF documents.