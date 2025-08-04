# Without File Upload option:

import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

# Title
st.set_page_config(page_title="RAG - Groq", layout="centered")
st.title("RAG Application - Powered by Grace College of Engineering 'Web-info'")

urls = [
    "https://grace.edu.in/index",
    "https://grace.edu.in/aboutus",
    "https://grace.edu.in/academic",
    "https://grace.edu.in/Admin",
    "https://grace.edu.in/helpdesk",
    "https://grace.edu.in/placement"
]

# --- Custom fallback scraper ---
def scrape_with_bs4(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        return f"Error scraping {url}: {e}"

# --- Try to get data ---
@st.cache_resource(show_spinner="📄 Loading and scraping documents...")
def load_documents():
    texts = [scrape_with_bs4(url) for url in urls]
    docs = [Document(page_content=txt) for txt in texts if txt.strip()]
    return docs

docs = load_documents()

# --- Show scraped content ---
if not docs or all(not doc.page_content.strip() for doc in docs):
    st.error(" No content was scraped from the provided URLs. Please check the URLs or your internet connection.")
    st.stop()

#  Show the first 500 characters of each scraped doc for debugging
with st.expander("Show scraped content (debug)"):
    for i, doc in enumerate(docs):
        st.write(f"**URL {i+1}:** {urls[i]}")
        st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

# --- Split text into chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

if not splits:
    st.error(" No text chunks were created from the scraped documents. Aborting.")
    st.stop()

# --- Embeddings ---
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(splits, embedding=embedding_model)


# Retrieve more chunks for richer context
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# --- Load LLM from Groq ---
llm = ChatGroq(
    temperature=0.4,
    model_name="gemma2-9b-it",  
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# --- Prompt template ---

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question in as much detail as possible. "
    "If you don't know the answer, say that you don't know. "
    "Provide a comprehensive, well-structured, and informative response, using all relevant information from the context.\n\n"
    "{context}"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# --- chat input ---
query = st.chat_input("Ask a question about Grace College:")

if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    with st.spinner("🔎 Processing your query..."):
        response = rag_chain.invoke({"input": query})
        st.success("✅ Answer:")
        st.write(response["answer"])




# import streamlit as st
# import os
# import time
# import requests
# from bs4 import BeautifulSoup
# from dotenv import load_dotenv
# from io import StringIO, BytesIO

# from langchain.docstore.document import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.prompts import ChatPromptTemplate
# from langchain_groq import ChatGroq

# from PyPDF2 import PdfReader

# load_dotenv()

# st.set_page_config(page_title="RAG - Groq", layout="centered")
# st.title("RAG Application - Powered by Grace College of Engineering")

# # --- Define URLs ---
# urls = [
#     "https://grace.edu.in/index",
#     "https://grace.edu.in/aboutus",
#     "https://grace.edu.in/academic"
# ]

# # --- Custom fallback scraper ---
# def scrape_with_bs4(url):
#     try:
#         res = requests.get(url, timeout=10)
#         soup = BeautifulSoup(res.text, "html.parser")
#         for script in soup(["script", "style"]):
#             script.decompose()
#         return soup.get_text(separator=" ", strip=True)
#     except Exception as e:
#         return f"Error scraping {url}: {e}"

# # --- Try to get data ---
# @st.cache_resource(show_spinner="📄 Loading and scraping documents...")
# def load_scraped_documents():
#     texts = [scrape_with_bs4(url) for url in urls]
#     docs = [Document(page_content=txt) for txt in texts if txt.strip()]
#     return docs

# # --- File upload ---
# uploaded_files = st.file_uploader("📁 Upload files (PDF, TXT)", type=['pdf', 'txt'], accept_multiple_files=True)

# def load_uploaded_documents(uploaded_files):
#     docs = []
#     for file in uploaded_files:
#         if file.type == "application/pdf":
#             pdf = PdfReader(file)
#             text = ""
#             for page in pdf.pages:
#                 text += page.extract_text() or ""
#             docs.append(Document(page_content=text))
#         elif file.type == "text/plain":
#             stringio = StringIO(file.getvalue().decode("utf-8"))
#             text = stringio.read()
#             docs.append(Document(page_content=text))
#     return docs

# scraped_docs = load_scraped_documents()
# uploaded_docs = load_uploaded_documents(uploaded_files) if uploaded_files else []

# # Combine both
# docs = scraped_docs + uploaded_docs

# if not docs or all(not doc.page_content.strip() for doc in docs):
#     st.error("No content found from scraping or uploaded files.")
#     st.stop()

# # Show the first 500 characters for debug
# with st.expander("🔍 Show loaded content (scraped + uploaded)"):
#     for i, doc in enumerate(docs):
#         st.write(f"**Doc {i+1}:**")
#         st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

# # --- Split text ---
# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = splitter.split_documents(docs)

# if not splits:
#     st.error("No text chunks created.")
#     st.stop()

# # --- Embeddings ---
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectorstore = Chroma.from_documents(splits, embedding=embedding_model)

# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# # --- Load Groq LLM ---
# llm = ChatGroq(
#     temperature=0.4,
#     model_name="gemma2-9b-it",
#     groq_api_key=os.getenv("GROQ_API_KEY")
# )

# # --- Prompt template ---
# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer the question in as much detail as possible. "
#     "If you don't know the answer, say that you don't know.\n\n"
#     "{context}"
# )

# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", system_prompt),
#     ("human", "{input}")
# ])

# # --- Chat input ---
# query = st.chat_input("Ask a question about Grace College or uploaded files:")

# if query:
#     question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
#     rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#     with st.spinner("🔎 Processing your query..."):
#         response = rag_chain.invoke({"input": query})
#         st.success("✅ Answer:")
#         st.write(response["answer"])










# import streamlit as st
# import time
# # from langchain_groq import ChatGroq
# from langchain_community.document_loaders import UnstructuredURLLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.prompts import ChatPromptTemplate
# from langchain.llms import OpenAI

# from dotenv import load_dotenv
# load_dotenv()

# st.title("RAG Application")

# urls = ['https://grace.edu.in/index','https://grace.edu.in/aboutus','https://grace.edu.in/academic']
# loader = UnstructuredURLLoader(urls=urls)
# data = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# docs = text_splitter.split_documents(data)

# all_splits = docs
# vectorstore = Chroma.from_documents(
#     documents=all_splits,
#     embedding=OpenAIEmbeddings()
# )
# retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": 6})

# retrieved_docs = retriever.invoke("What is the mission of Grace College?")

# llm = OpenAI(temperature=0.4, max_tokens=500)

# query = st.chat_input("Ask a question about Grace College:")
# prompt = query

# system_prompt =(
#      "You are an assistant for question-aswering tasks. "
#     "Use the following pieces of retrieved context to answer "
#     "the question. If you don't know the answer, say that you "
#     "don't know. Use three sentences maximum and keep the "
#     "answer concise."
#     "\n\n"
#     "{context}"
# )

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# if query:
#     question_answer_chain = create_stuff_documents_chain(llm, prompt)
#     rag_chain = create_retrieval_chain(retriever, question_answer_chain)
#     with st.spinner("Processing your query..."):
#         time.sleep(2)
#         response = rag_chain.invoke({"input": query})
#         st.write(response['answer'])