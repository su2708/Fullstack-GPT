import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📖",
)

st.title("DocumentGPT")

st.markdown(
    """
    Welcome!
    
    Use this chatbot to ask questions to an AI about your files!
    """
)

file = st.file_uploader("Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"])

if file:
    st.write(file)
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    st.write(file_content, file_path)