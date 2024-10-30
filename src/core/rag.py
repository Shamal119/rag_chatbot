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

        # Updated prompt template for precise and clear answers
        self.qa_template = """
        Answer based on the context provided. Be concise and specific. If the answer cannot be found in the context, say so.

        Context: {context}
        Chat History: {chat_history}
        Question: {question}

        Answer: Let me provide the information from the context:"""

    def create_vectorstore(self, documents):
        texts = self.text_splitter.split_documents(documents)
        return FAISS.from_documents(texts, self.embeddings)

    def get_chain(self, vectorstore):
        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=self.qa_template
        )
        
        # Chain set to return only answer, ensuring compatibility with memory
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=False,  # Only return 'answer' for memory compatibility
            memory=self.memory,
            output_key="answer",
            verbose=True
        )
        return chain

