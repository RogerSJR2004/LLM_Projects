
# === accumulate all docs from all sources in session state, deduplicate, and use for retrieval ===
import streamlit as st
import os
import time
import asyncio
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from dotenv import load_dotenv

load_dotenv()

try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

st.set_page_config(page_title="RAG App", layout="centered")

# --- Session state for accumulating all docs ---
if "all_docs" not in st.session_state:
    st.session_state.all_docs = []

# Sidebar Inputs
st.sidebar.title("📂 Upload & URL Input")
uploaded_file = st.sidebar.file_uploader("Upload PDF File", type=["pdf"])
url = st.sidebar.text_input("Enter a Webpage URL")

# Add new docs to session state (accumulate)
new_docs = []
if uploaded_file:
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())
    loader = PyPDFLoader("temp_uploaded.pdf")
    new_docs.extend(loader.load())
    os.remove("temp_uploaded.pdf")
if url:
    loader = WebBaseLoader(url)
    new_docs.extend(loader.load())

# Only add new docs if they are not already present (deduplicate by content hash)
def doc_hash(doc):
    import hashlib
    return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()

existing_hashes = set(doc_hash(doc) for doc in st.session_state.all_docs)
for doc in new_docs:
    if doc_hash(doc) not in existing_hashes:
        st.session_state.all_docs.append(doc)
        existing_hashes.add(doc_hash(doc))

# Text Splitting and RAG setup
if st.session_state.all_docs:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(st.session_state.all_docs)

    # Deduplicate split_docs by content hash
    seen_chunks = set()
    unique_split_docs = []
    for doc in split_docs:
        h = doc_hash(doc)
        if h not in seen_chunks:
            unique_split_docs.append(doc)
            seen_chunks.add(h)

    vectorstore = Chroma.from_documents(
        documents=unique_split_docs,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

    # LLM Setup
    llm = ChatGroq(
        temperature=0.4,
        model_name="gemma2-9b-it",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following retrieved context to answer the question clearly and helpfully. "
        "If you don't know the answer, just say so. Keep it concise and based on the context.\n\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Chat Interface
    st.title("RAG Chat Assistant")

    query = st.chat_input("Ask anything from the document or URL...")

    if query:
        with st.spinner("Thinking..."):
            time.sleep(1)
        response = rag_chain.invoke({"input": query})
        st.chat_message("assistant").write(response['answer'])
        with st.expander("🔍 View Context Used"):
            for doc in response['context']:
                st.markdown(f"- {doc.page_content[:300]}...")
else:
    st.info("Upload a PDF or enter a URL from the sidebar to begin.")




##===== Code Model 2 ======== :

# import streamlit as st
# import os
# import time
# import asyncio

# from langchain.docstore.document import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.prompts import ChatPromptTemplate
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import PyPDFLoader

# from dotenv import load_dotenv
# load_dotenv()

# # Ensure the event loop is set
# try:
#     asyncio.get_event_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())

# st.set_page_config(page_title="RAG - Groq", layout="centered")
# st.title("RAG Application - Powered by IEE-Paper")

# loader = PyPDFLoader("Agentic_AI.pdf")
# data = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# docs = text_splitter.split_documents(data)

# vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# llm = ChatGroq(
#     temperature=0.4,
#     model_name="gemma2-9b-it",  
#     groq_api_key=os.getenv("GROQ_API_KEY")
# )

# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer the question in as much detail as possible. "
#     "If you don't know the answer, say that you don't know. "
#     "Use the context to answer the question, but do not repeat the context verbatim."
#      "answer consise."
#     "\n\n"
#     "{context}"
# )


# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# query = st.chat_input("Ask a question about Autonoums AI")

# if query:
#     with st.spinner("Retrieving information..."):
#         time.sleep(1)
#     question_answer_chain = create_stuff_documents_chain(llm, prompt)
#     rag_chain = create_retrieval_chain(retriever, question_answer_chain)
#     response = rag_chain.invoke({"input": query})
#     st.chat_message("assistant").write(response['answer'])  # Display the output of the RAG chain
#     st.chat_message("context").write(response['context'])  # Display the context used for the answer

