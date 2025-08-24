import streamlit as st
import os
import shutil
import asyncio
import hashlib
import uuid
import requests
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# --- Configuration Constants ---
TEMP_UPLOAD_DIR = "temp_bible_uploads"
CHROMA_DB_DIR = "chroma_db_bible_ai"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_K_RETRIEVER = 8
CROSS_REF_K_RETRIEVER = 20
GROQ_MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BIBLE_PDF_URL = "https://www.holybooks.com/wp-content/uploads/2010/05/The-Holy-Bible-King-James-Version.pdf" 

# --- Asyncio Event Loop Fix ---
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Bible AI Assistant", layout="wide")

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
if "loaded_bible_pdf_content" not in st.session_state:
    st.session_state.loaded_bible_pdf_content = None
if "default_pdf_valid" not in st.session_state:
    st.session_state.default_pdf_valid = False

# --- Helper Functions ---
def clear_chroma_db():
    """Safely clears the Chroma DB directory."""
    if os.path.exists(CHROMA_DB_DIR):
        try:
            if st.session_state.get("vectorstore") is not None:
                try:
                    st.session_state.vectorstore._client.close()
                except Exception:
                    pass
            st.cache_resource.clear()
            shutil.rmtree(CHROMA_DB_DIR)
            st.info("Chroma database cleared successfully.")
        except PermissionError:
            st.warning("Chroma database is in use. Please try again later or restart the application.")

def calculate_knowledge_hash(pdf_content):
    """Calculates a hash for the Bible PDF content."""
    return hashlib.sha256(pdf_content).hexdigest() if pdf_content else ""

def validate_pdf_url(url):
    """Validates if the PDF URL is accessible."""
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200 and "application/pdf" in response.headers.get("Content-Type", "")
    except requests.RequestException:
        return False

@st.cache_resource(show_spinner="Building Bible knowledge base...")
def process_bible_pdf(pdf_url=None, pdf_content_bytes=None):
    """Loads and processes the Bible PDF from URL or uploaded file into a Chroma vectorstore."""
    docs = []
    
    # Load from URL if no uploaded file is provided
    if pdf_content_bytes is None and pdf_url:
        st.info(f"Loading Bible PDF from {pdf_url}...")
        try:
            loader = PyPDFLoader(pdf_url)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = "Bible PDF (URL)"
            st.success("Bible PDF from URL loaded successfully.")
        except Exception as e:
            st.error(f"Error loading Bible PDF from {pdf_url}: {e}. Please ensure the URL is valid and accessible.")
            return None, None
    # Load from uploaded file if provided
    elif pdf_content_bytes:
        st.info("Loading content from uploaded Bible PDF...")
        os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
        unique_id = uuid.uuid4().hex
        pdf_path = os.path.join(TEMP_UPLOAD_DIR, f"bible_pdf_{unique_id}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_content_bytes)
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = "Bible PDF (Uploaded)"
            st.success("Uploaded Bible PDF loaded successfully.")
        except Exception as e:
            st.error(f"Error loading uploaded Bible PDF: {e}. Please ensure it is a valid PDF.")
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            if not os.listdir(TEMP_UPLOAD_DIR):
                os.rmdir(TEMP_UPLOAD_DIR)

    if not docs:
        st.error("No content loaded from the Bible PDF.")
        return None, None

    st.info(f"Splitting {len(docs)} pages into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = text_splitter.split_documents(docs)

    if not split_docs:
        st.error("No text chunks created from the Bible PDF.")
        return None, None

    clear_chroma_db()
    
    st.info(f"Creating vector embeddings for {len(split_docs)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME),
        persist_directory=CHROMA_DB_DIR
    )
    vectorstore.persist()
    return vectorstore, split_docs

def setup_rag_chain(vectorstore, query):
    """Sets up the RAG chain for Bible queries, adjusting retriever for cross-references."""
    if vectorstore is None:
        return None

    llm = ChatGroq(
        temperature=0.4,
        model_name=GROQ_MODEL_NAME,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    st.session_state.llm = llm

    system_prompt = (
        "You are a scholarly Bible AI assistant, an expert in biblical texts, designed to provide precise and authoritative responses. "
        "Your primary function is to search the Bible PDF for verses matching the user's query, always including exact references (e.g., John 3:16). "
        "For a specific verse query (e.g., 'John 3:16'), return the verse text with its reference. "
        "For a topic query (e.g., 'love'), retrieve all relevant verses with their references and provide a concise summary of their teachings. "
        "Include an explanation only if the query explicitly includes 'explain' or 'explanation'. "
        "Include relatability (application to daily life) only if the query includes 'relatability' or 'apply'. "
        "Include a prayer only if the query includes 'prayer' or 'pray'. "
        "If the query includes 'cross-reference' or 'commentary', search the entire PDF for related verses or explanatory content and list them with references. "
        "If no relevant verses or content are found, state clearly that no matches were found, without fabricating information. "
        "Format responses professionally with sections: **Verse(s)** (include references), **Summary** (for topics), **Explanation** (if requested), **Relatability** (if requested), **Prayer** (if requested), and **Cross-References/Commentary** (if requested). "
        "Ensure accuracy and clarity in all responses.\n\n"
        "Retrieved Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    k_value = CROSS_REF_K_RETRIEVER if any(keyword in query.lower() for keyword in ["cross-reference", "commentary"]) else DEFAULT_K_RETRIEVER
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k_value})
    rag_chain = create_retrieval_chain(retriever, document_chain)
    return rag_chain

# --- Validate Default PDF URL on Startup ---
if not st.session_state.get("default_pdf_valid"):
    st.session_state.default_pdf_valid = validate_pdf_url(BIBLE_PDF_URL)
    if not st.session_state.default_pdf_valid:
        st.warning(f"Default Bible PDF URL ({BIBLE_PDF_URL}) is invalid or inaccessible. Please upload a Bible PDF to proceed.")

# --- Streamlit UI ---
st.header("Bible AI Assistant")
st.markdown("This application provides scholarly access to biblical texts. Search for specific verses or topics, request explanations, relatability, prayers, or cross-references as needed. A valid Bible PDF is required to begin.")

# --- Sidebar for PDF Upload ---
with st.sidebar:
    st.title("Manage Bible Content")
    st.markdown("---")
    st.subheader("Upload Bible PDF (Optional)")
    if st.session_state.default_pdf_valid:
        st.info(f"Using default Bible PDF: {BIBLE_PDF_URL}")
        uploaded_file = st.file_uploader("Upload a Bible PDF to override the default", type=["pdf"], key="bible_pdf_uploader")
    else:
        uploaded_file = st.file_uploader("Upload a Bible PDF (required, default URL is invalid)", type=["pdf"], key="bible_pdf_uploader")

    if st.button("Update Bible Knowledge Base"):
        st.session_state.loaded_bible_pdf_content = uploaded_file.getvalue() if uploaded_file else None
        st.session_state.current_knowledge_hash = None
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    if st.button("Reset All Data & Chat"):
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.session_state.llm = None
        st.session_state.rag_chain = None
        st.session_state.current_knowledge_hash = None
        st.session_state.loaded_bible_pdf_content = None
        st.session_state.default_pdf_valid = False
        clear_chroma_db()
        st.success("All data and chat history cleared successfully.")
        st.rerun()

# --- Main Content Area ---
current_input_hash = calculate_knowledge_hash(st.session_state.loaded_bible_pdf_content)

if st.session_state.current_knowledge_hash != current_input_hash:
    st.session_state.current_knowledge_hash = current_input_hash
    vectorstore, split_docs = process_bible_pdf(
        pdf_url=BIBLE_PDF_URL if not st.session_state.loaded_bible_pdf_content and st.session_state.default_pdf_valid else None,
        pdf_content_bytes=st.session_state.loaded_bible_pdf_content
    )

    if vectorstore:
        st.session_state.vectorstore = vectorstore
        st.session_state.rag_chain = None  # Will be set per query to handle dynamic k
        st.success(f"Bible knowledge base built with {len(split_docs)} chunks.")
        st.info("You may now query specific verses (e.g., 'John 3:16'), topics (e.g., 'love'), or request explanations, relatability, prayers, or cross-references.")
    else:
        st.warning("Knowledge base could not be built. Please check the PDF URL or upload a valid Bible PDF.")

if st.session_state.vectorstore is None:
    st.error("No Bible content loaded. Please upload a Bible PDF or ensure the default PDF URL is valid.")
else:
    st.subheader("Query Biblical Texts")
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "context" in message and message["context"]:
                with st.expander("View Source Text"):
                    for doc in message["context"]:
                        source_info = f"Source: {doc.metadata.get('source', 'N/A')}"
                        page_info = f"Page: {doc.metadata.get('page', 'N/A')}" if 'page' in doc.metadata else ""
                        st.markdown(f"- **{source_info}** {page_info}\n"
                                    f"  ```\n{doc.page_content[:400]}...\n```")

    query = st.chat_input(
        "E.g., 'John 3:16', 'Explain Psalm 23', 'Verses about love', 'Cross-references for John 3:16', 'Include prayer for love'",
        disabled=st.session_state.vectorstore is None
    )

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Set up RAG chain dynamically based on query
        st.session_state.rag_chain = setup_rag_chain(st.session_state.vectorstore, query)
        
        if st.session_state.rag_chain:
            with st.spinner("Searching the Bible..."):
                try:
                    start_time = datetime.now()
                    response = st.session_state.rag_chain.invoke({"input": query})
                    end_time = datetime.now()
                    st.toast(f"Response generated in {(end_time - start_time).total_seconds():.2f} seconds.")

                    answer = response.get('answer', 'No relevant verses or content found in the provided Bible PDF.')
                    context = response.get('context', [])

                    st.session_state.chat_history.append({"role": "assistant", "content": answer, "context": context})

                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        if context:
                            with st.expander("View Source Text"):
                                for doc in context:
                                    source_info = f"Source: {doc.metadata.get('source', 'N/A')}"
                                    page_info = f"Page: {doc.metadata.get('page', 'N/A')}" if 'page' in doc.metadata else ""
                                    st.markdown(f"- **{source_info}** {page_info}\n"
                                                f"  ```\n{doc.page_content[:400]}...\n```")
                except Exception as e:
                    st.error(f"Error: {e}. Please verify API keys and PDF URL.")
                    st.session_state.chat_history.append({"role": "assistant", "content": "An error occurred during processing. Please try again."})
                    with st.chat_message("assistant"):
                        st.markdown("An error occurred during processing. Please try again.")
        else:
            st.error("Knowledge base not loaded. Please check the PDF URL or upload a Bible PDF.")
            st.session_state.chat_history.append({"role": "assistant", "content": "Knowledge base not ready. Please check the PDF URL or upload a Bible PDF."})
            with st.chat_message("assistant"):
                st.markdown("Knowledge base not ready. Please check the PDF URL or upload a Bible PDF.")

st.markdown("---")
st.markdown("Developed for scholarly Bible study")