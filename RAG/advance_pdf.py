import streamlit as st
import os
import shutil
import asyncio
import hashlib
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import uuid

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

# --- Configuration Constants ---
TEMP_UPLOAD_DIR = "temp_rag_uploads"
CHROMA_DB_DIR = "chroma_db_unified_ai" 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_K_RETRIEVER = 8
GROQ_MODEL_NAME = "gemma2-9b-it"
EMBEDDING_MODEL_NAME = "models/embedding-001"

# Pre-defined URLs for AI-related knowledge sources
AI_RELATED_URLS = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://www.ibm.com/topics/artificial-intelligence",
    "https://www.geeksforgeeks.org/artificial-intelligence/artificial-intelligence/",
    "https://www.geeksforgeeks.org/machine-learning/search-algorithms-in-ai/"
]

# --- Asyncio Event Loop Fix ---
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="IntelliStudy: AI Research Assistant", layout="wide")

# --- Initialize Session State Variables ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "llm" not in st.session_state:
    st.session_state.llm = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "current_knowledge_hash" not in st.session_state:
    st.session_state.current_knowledge_hash = None
if "loaded_user_pdf_content" not in st.session_state:
    st.session_state.loaded_user_pdf_content = None
if "loaded_user_url" not in st.session_state:
    st.session_state.loaded_user_url = None

# --- Helper Functions ---

def clear_chroma_db():
    """Safely clears the Chroma DB directory."""
    if os.path.exists(CHROMA_DB_DIR):
        try:
            if st.session_state.get("vectorstore") is not None:
                try:
                    st.session_state.vectorstore._client.close()
                except Exception:
                    pass  # Ignore if already closed or not available
            st.cache_resource.clear() # Clear cache for process_all_documents
            shutil.rmtree(CHROMA_DB_DIR)
            st.info("Chroma DB cleared.")
        except PermissionError:
            st.warning("Chroma DB is currently in use. Please try again after a few seconds or restart the app.")

def calculate_knowledge_hash(user_pdf_content, user_url):
    """Calculates a combined hash for all knowledge sources."""
    pdf_hash = hashlib.sha256(user_pdf_content).hexdigest() if user_pdf_content else ""
    url_hash = hashlib.sha256(user_url.encode('utf-8')).hexdigest() if user_url else ""
    # Incorporate the new AI_RELATED_URLS into the hash
    ai_urls_string = "-".join(sorted(AI_RELATED_URLS)) # Sort for consistent hash
    return hashlib.sha256(f"{pdf_hash}-{url_hash}-{ai_urls_string}-v1.1".encode('utf-8')).hexdigest() 

def scrape_with_bs4_fallback(url):
    """Custom BS4 scraper as a fallback if WebBaseLoader struggles."""
    try:
        res = requests.get(url, timeout=15)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text if text else None
    except requests.exceptions.RequestException as e:
        st.warning(f"Warning: Could not scrape '{url}' using requests (Error: {e}). Skipping.")
        return None
    except Exception as e:
        st.warning(f"Warning: Error during BS4 parsing of '{url}' (Error: {e}). Skipping.")
        return None

@st.cache_resource(show_spinner="🧠 Building your AI knowledge base (this may take a moment!)...")
def process_all_documents(user_pdf_content_bytes, user_url_str):
    """
    Loads documents from predefined AI URLs, user-uploaded PDF, and user-provided URL.
    Splits them and creates a unified Chroma vectorstore.
    """
    all_docs = []
    
    # 1. Load Pre-defined AI-Related URLs
    st.info("Loading pre-built knowledge from AI-related websites...")
    for url in AI_RELATED_URLS:
        try:
            loader = WebBaseLoader(url)
            docs_from_url = loader.load()
            for doc in docs_from_url: # Add metadata for source tracking
                 doc.metadata["source"] = url
                 all_docs.append(doc)
            st.success(f"Successfully loaded '{url}' with WebBaseLoader.")
        except Exception as e:
            st.warning(f"Could not load '{url}' with WebBaseLoader (Error: {e}). Attempting fallback scraper...")
            scraped_text = scrape_with_bs4_fallback(url)
            if scraped_text:
                all_docs.append(Document(page_content=scraped_text, metadata={"source": url, "loader": "bs4_fallback"}))
                st.success(f"Successfully loaded '{url}' with BS4 fallback.")
            else:
                st.error(f"Failed to get content from '{url}' even with fallback. Skipping this URL.")

    # 2. Load User-Uploaded PDF
    if user_pdf_content_bytes:
        st.info("Loading content from your uploaded PDF...")
        os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
        unique_id = uuid.uuid4().hex
        pdf_path = os.path.join(TEMP_UPLOAD_DIR, f"user_uploaded_pdf_{unique_id}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(user_pdf_content_bytes)
        try:
            loader = PyPDFLoader(pdf_path)
            docs_from_pdf = loader.load()
            for doc in docs_from_pdf: # Add metadata for source tracking
                 doc.metadata["source"] = "User Uploaded PDF"
                 all_docs.append(doc)
            st.success("PDF loaded successfully.")
        except Exception as e:
            st.error(f"Error loading uploaded PDF: {e}. Please ensure it's a valid PDF.")
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            if not os.listdir(TEMP_UPLOAD_DIR):
                os.rmdir(TEMP_UPLOAD_DIR)

    # 3. Load User-Provided URL
    if user_url_str:
        st.info(f"Loading content from user-provided URL: {user_url_str}...")
        try:
            loader = WebBaseLoader(user_url_str)
            docs_from_url_input = loader.load()
            for doc in docs_from_url_input: # Add metadata for source tracking
                 doc.metadata["source"] = user_url_str
                 all_docs.append(doc)
            st.success("User URL content loaded successfully.")
        except Exception as e:
            st.error(f"Error loading user URL '{user_url_str}': {e}. Please check the URL.")

    if not all_docs:
        st.error("No documents could be loaded from any source. Please check inputs and try again.")
        return None, None

    st.info(f"Splitting {len(all_docs)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = text_splitter.split_documents(all_docs)

    if not split_docs:
        st.error("No text chunks were created from the loaded documents. Aborting.")
        return None, None

    clear_chroma_db()
    
    st.info(f"Creating vector embeddings for {len(split_docs)} chunks and building knowledge base...")
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME),
        persist_directory=CHROMA_DB_DIR
    )
    vectorstore.persist()

    return vectorstore, split_docs

def setup_rag_chain(vectorstore):
    """
    Sets up the RAG chain using the provided vectorstore and Groq LLM.
    """
    if vectorstore is None:
        return None

    llm = ChatGroq(
        temperature=0.4,
        model_name=GROQ_MODEL_NAME,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    st.session_state.llm = llm

    system_prompt = (
        "You are an intelligent AI research assistant. "
        "Use the following retrieved context to answer the question clearly, comprehensively, and helpfully. "
        "Provide as much detail as possible based on the context. "
        "If you don't know the answer or the information is not in the provided context, "
        "please state that you cannot find the answer. Do not make up information. "
        "Structure your response well for readability, using bullet points or paragraphs as appropriate.\n\n"
        "Retrieved Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    Youtube_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": DEFAULT_K_RETRIEVER})
    rag_chain = create_retrieval_chain(retriever, Youtube_chain)
    return rag_chain

# --- Streamlit UI ---
st.header("🎓 IntelliStudy: AI Research Assistant")
st.markdown("This assistant can answer questions from pre-loaded AI knowledge, plus any PDFs or URLs you provide.")

# --- Sidebar for Knowledge Inputs ---
with st.sidebar:
    st.title("Manage Your Knowledge Base")
    st.markdown("---")

    st.subheader("1. Add Personal Documents")
    uploaded_file = st.file_uploader("Upload PDF File (e.g., research papers)", type=["pdf"], key="user_pdf_uploader")
    user_url = st.text_input("Enter a Webpage URL (e.g., specific AI article link)", key="user_web_url")

    if st.button("Update Knowledge Base", help="Click to process new PDF/URL additions and rebuild the knowledge base."):
        st.session_state.loaded_user_pdf_content = uploaded_file.getvalue() if uploaded_file else None
        st.session_state.loaded_user_url = user_url
        st.session_state.current_knowledge_hash = None # Force re-processing
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.subheader("2. Pre-built Knowledge")
    st.info("General AI-related information from popular websites is pre-loaded by default.")
    with st.expander("View Pre-loaded AI URLs"):
        for url in AI_RELATED_URLS:
            st.markdown(f"- [{url.split('//')[-1].split('/')[0]}...]{url})") # Shorter display

    st.markdown("---")
    if st.button("Reset All Data & Chat", help="Clears all loaded documents, vector store, and chat history."):
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.session_state.llm = None
        st.session_state.rag_chain = None
        st.session_state.current_knowledge_hash = None
        st.session_state.loaded_user_pdf_content = None
        st.session_state.loaded_user_url = None
        clear_chroma_db()
        st.success("All data and chat history cleared!")
        st.rerun()

# --- Main Content Area ---
current_input_hash = calculate_knowledge_hash(
    st.session_state.loaded_user_pdf_content,
    st.session_state.loaded_user_url
)

if st.session_state.current_knowledge_hash != current_input_hash:
    st.session_state.current_knowledge_hash = current_input_hash
    vectorstore, split_docs = process_all_documents(
        st.session_state.loaded_user_pdf_content,
        st.session_state.loaded_user_url
    )

    if vectorstore:
        st.session_state.vectorstore = vectorstore
        st.session_state.rag_chain = setup_rag_chain(vectorstore)
        st.success(f"AI Knowledge base successfully built with {len(split_docs)} chunks from all sources!")
        st.info("You can now ask questions about AI or your added documents.")
    else:
        st.warning("Knowledge base could not be built. Please try uploading valid files/URLs.")
        st.session_state.vectorstore = None
        st.session_state.rag_chain = None

if st.session_state.vectorstore is None:
    st.warning("Knowledge base not ready. Please use the sidebar to load documents or click 'Update Knowledge Base' if you've provided inputs.")
else:
    st.subheader(" Ask Anything about AI or Your Documents")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "context" in message and message["context"]:
                with st.expander("🔍 View Context Used"):
                    for doc in message["context"]:
                        source_info = f"Source: {doc.metadata.get('source', 'N/A')}"
                        page_info = f"Page: {doc.metadata.get('page', 'N/A')}" if 'page' in doc.metadata else ""
                        loader_info = f"Loader: {doc.metadata.get('loader', 'N/A')}" if 'loader' in doc.metadata else ""
                        st.markdown(f"- **{source_info}** {page_info} {loader_info}\n"
                                    f"  ```\n{doc.page_content[:400]}...\n```")

    query = st.chat_input("E.g., 'What is machine learning?', 'Explain neural networks from my PDF', 'What is reinforcement learning?'")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        if st.session_state.rag_chain:
            with st.spinner("Thinking..."):
                try:
                    start_time = datetime.now()
                    response = st.session_state.rag_chain.invoke({"input": query})
                    end_time = datetime.now()
                    st.toast(f"Response generated in {(end_time - start_time).total_seconds():.2f} seconds.")

                    answer = response.get('answer', 'Sorry, I could not find an answer in the provided context.')
                    context = response.get('context', [])

                    st.session_state.chat_history.append({"role": "assistant", "content": answer, "context": context})

                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        if context:
                            with st.expander("🔍 View Context Used"):
                                for doc in context:
                                    source_info = f"Source: {doc.metadata.get('source', 'N/A')}"
                                    page_info = f"Page: {doc.metadata.get('page', 'N/A')}" if 'page' in doc.metadata else ""
                                    loader_info = f"Loader: {doc.metadata.get('loader', 'N/A')}" if 'loader' in doc.metadata else ""
                                    st.markdown(f"- **{source_info}** {page_info} {loader_info}\n"
                                                f"  ```\n{doc.page_content[:400]}...\n```")
                except Exception as e:
                    st.error(f"An error occurred during response generation: {e}. Please ensure your API keys are correct.")
                    st.session_state.chat_history.append({"role": "assistant", "content": "An error occurred while processing your request. Please try again."})
                    with st.chat_message("assistant"):
                        st.markdown("An error occurred while processing your request. Please try again.")
        else:
            st.warning("Knowledge base not fully loaded. Please ensure documents are processed.")
            st.session_state.chat_history.append({"role": "assistant", "content": "Knowledge base not ready. Please load documents first."})
            with st.chat_message("assistant"):
                st.markdown("Knowledge base not ready. Please load documents first.")

st.markdown("---")
st.markdown("Developed with ❤️ for Students - IntelliStudy App")