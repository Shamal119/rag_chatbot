from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def get_llm_and_embeddings(api_choice, api_key):
    if api_choice == "OpenAI":
        llm = ChatOpenAI(temperature=0, api_key=api_key)
        embeddings = OpenAIEmbeddings(api_key=api_key)
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-flash", google_api_key=api_key)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    return llm, embeddings