import streamlit as st
import os
import time

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="RAG - Groq", layout="centered")
st.title("RAG Application - Powered by IEE-Paper")

loader = PyPDFLoader("Agentic_AI.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

llm = ChatGroq(
    temperature=0.4,
    model_name="gemma2-9b-it",  
    groq_api_key=os.getenv("GROQ_API_KEY")
)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question in as much detail as possible. "
    "If you don't know the answer, say that you don't know. "
    "Use the context to answer the question, but do not repeat the context verbatim."
     "answer consise."
    "\n\n"
    "{context}"
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

query = st.chat_input("Ask a question about Autonoums AI")

if query:
    with st.spinner("Retrieving information..."):
        time.sleep(1)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"question": query})
    st.chat_message(response)
    st.write(response['answer'])  # Display the output of the RAG chain

    st.chat_message("context")
    st.write(response['context'])  # Display the context used for the answer