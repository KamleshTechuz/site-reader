import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import requests
from bs4 import BeautifulSoup
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from urllib.parse import urlparse

st.cache_data.clear()
st.cache_resource.clear()

def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def extract_base_url(url):
    parsed_url = urlparse(url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}"

def vector_embedding(urls):
    st.session_state.loader = WebBaseLoader(web_paths=urls)
    st.session_state.documents = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.doc_text = st.session_state.text_splitter.split_documents(st.session_state.documents)
    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.vector = FAISS.from_documents(st.session_state.doc_text, st.session_state.embeddings)

def fetch_sitemap(url):
    response = requests.get(url)
    if response.status_code == 200:
        sitemap_content = response.content
        soup = BeautifulSoup(sitemap_content, 'xml')
        urls = [loc.text.strip() for loc in soup.find_all('loc')]
        return urls
    else:
        return []

st.title("SiteGPT")

sitemap_url_input = st.text_input("Enter the web site URL here")

if sitemap_url_input:
    if validate_url(sitemap_url_input):
        sitemap_url = extract_base_url(sitemap_url_input)
    else:
        st.error("Invalid URL. Please enter a valid website URL.")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
""")

llm = ChatGroq(model="llama3-8b-8192")

if 'sitemap_url' in locals() and st.button("Fetch Sitemap and Create Embeddings"):
    url = f'{sitemap_url}/sitemap.xml'
    sitemap_data = fetch_sitemap(url)

    print('sitemap_data : ',sitemap_data)
    
    if sitemap_data:
        vector_embedding(sitemap_data)
        st.write("Vector Store BD is available")
    else:
        st.error("Failed to fetch sitemap. Please ensure the website has a sitemap.xml.")

query = st.text_input("You can ask anything...")

if query and st.button("Submit Query"):
    if "vector" in st.session_state:
        doc_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vector.as_retriever()
        retriever_chain = create_retrieval_chain(retriever, doc_chain)
        response = retriever_chain.invoke({"input": query})
        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.error("Vector Store is not available. Please fetch sitemap and create embeddings first.")
