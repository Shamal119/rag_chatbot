from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


class RAGSystem:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        # Humanized and detailed prompt template
        self.qa_template = """
        You are a friendly and helpful assistant. Based on the documents provided, answer the user's question in a clear and detailed manner.
        If the answer is in the context, start with a friendly tone. If you can't find the answer, just say that you couldn't find the information in the documents.

        Context:
        {context}

        Chat History:
        {chat_history}

        Question:
        {question}

        Helpful Answer:"""

    def create_vectorstore(self, documents):
        texts = self.text_splitter.split_documents(documents)
        return FAISS.from_documents(texts, self.embeddings)

    def get_chain(self, vectorstore):
        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=self.qa_template
        )
        
        # Chain now returns source documents and the answer
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,  # Return source documents
            memory=self.memory,
            verbose=True
        )
        return chain

