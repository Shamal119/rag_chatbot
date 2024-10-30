# src/utils/file_handler.py
import PyPDF2
import io

def process_file(uploaded_file):
    text = ""
    if uploaded_file.name.lower().endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    else:
        text = uploaded_file.getvalue().decode('utf-8')
    
    from langchain.schema import Document
    return [Document(page_content=text, metadata={"source": uploaded_file.name})]