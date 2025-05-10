from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

def create_vector_embeddings(data_path):
    if "vector" not in st.session_state:
        embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
        loader = PyPDFDirectoryLoader(data_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        vectors = FAISS.from_documents(final_documents, embedding=embeddings)
        return vectors

def get_results(input, vectors, ans_type):
    llm = ChatOllama(
        model="gemma3:1b",
        temperature=0.5,
    )

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on context provided.
        Provide the most relevant response based on context and give the answer.
        <context>
        {context}
        <context>
        Question: {input}
        Answer Type: {ans_type}
        """
    )
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, doc_chain)
    response = retriever_chain.invoke({"input": input, "ans_type": ans_type})

    return response



