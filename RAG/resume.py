import streamlit as st
import os
import shutil
import asyncio
from datetime import datetime
import hashlib # For robust file hash calculation

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# Load environment variables (API keys) from .env file
load_dotenv()

# --- Configuration Constants ---
# Define constants for easy modification and clarity
RESUME_TEMP_DIR = "temp_resume_uploads"  # Directory for temporary uploaded resume files
CHROMA_DB_DIR = "chroma_db_resume"      # Directory to persist Chroma vector database
CHUNK_SIZE = 1000                       # Size of text chunks for splitting documents
CHUNK_OVERLAP = 200                     # Overlap between consecutive text chunks
DEFAULT_K_RETRIEVER = 5                 # Number of top relevant chunks to retrieve from vectorstore
GROQ_MODEL_NAME = "gemma2-9b-it"        # Groq LLM model to use for generation
EMBEDDING_MODEL_NAME = "models/embedding-001" # Google Generative AI Embedding model

# --- Asyncio Event Loop Fix ---
# This block fixes a common RuntimeError in some environments (like Streamlit Cloud)
# where the asyncio event loop might already be running or not properly set.
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- Streamlit Page Configuration ---
# Sets up the Streamlit page title, layout, and icon for a professional look.
st.set_page_config(page_title="ResumeGenie: AI Resume Analyzer", layout="centered")

# --- Initialize Session State Variables ---
# Streamlit reruns the script on every interaction. Session state ensures data
# like the vectorstore, chat history, and LLM instances persist across reruns.
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "llm" not in st.session_state:
    st.session_state.llm = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "current_file_hash" not in st.session_state:
    st.session_state.current_file_hash = None # Tracks the uploaded file's content hash for caching

# --- Helper Functions ---

def clear_chroma_db(db_dir):
    """
    Safely clears the specified Chroma DB directory.
    This ensures a fresh knowledge base when a new resume is uploaded.
    """
    if os.path.exists(db_dir):
        try:
            if st.session_state.get("vectorstore") is not None:
                try:
                    st.session_state.vectorstore._client.close()
                except Exception:
                    pass  # Ignore if already closed or not available
            st.cache_resource.clear() # Clear st.cache_resource for process_resume if DB is cleared
            shutil.rmtree(db_dir)
            st.info(f"Cleared existing knowledge base at '{db_dir}'.")
        except PermissionError:
            st.warning("Chroma DB is currently in use. Please try again after a few seconds or restart the app.")

def calculate_file_hash(uploaded_file):
    """Calculates a SHA256 hash of the uploaded file's content."""
    if uploaded_file is None:
        return None
    file_bytes = uploaded_file.getvalue()
    return hashlib.sha256(file_bytes).hexdigest()

@st.cache_resource(show_spinner="Analyzing your resume... This might take a moment!")
def process_resume(uploaded_file_content):
    """
    Loads a PDF resume, splits it into chunks, and creates a Chroma vectorstore.
    This function is memoized by `st.cache_resource` to prevent reprocessing
    the same resume content multiple times.
    """
    if not uploaded_file_content:
        return None, None

    # Ensure temporary directory exists
    os.makedirs(RESUME_TEMP_DIR, exist_ok=True)
    
    # Create a temporary PDF file to be loaded by PyPDFLoader
    # Using a timestamped name to avoid potential conflicts in certain deployment scenarios
    temp_pdf_path = os.path.join(RESUME_TEMP_DIR, f"resume_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.pdf")
    
    try:
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file_content)
        
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load() # Load documents from the PDF
        
    except Exception as e:
        st.error(f"Error loading PDF: {e}. Please ensure it's a valid, text-searchable PDF.")
        return None, None
    finally:
        # Clean up the temporary PDF file immediately after loading
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        # Remove the temporary directory if it becomes empty
        if not os.listdir(RESUME_TEMP_DIR):
            os.rmdir(RESUME_TEMP_DIR)

    if not docs:
        st.warning("Could not extract any readable text from the uploaded PDF. "
                   "Ensure it's not a scanned image and is text-searchable.")
        return None, None

    # Split documents into smaller, manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = text_splitter.split_documents(docs)

    # Clear existing Chroma DB before creating a new one to prevent mixing data
    clear_chroma_db(CHROMA_DB_DIR)
    
    # Create Google Generative AI Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    # Create and persist the Chroma vectorstore from the split documents and embeddings
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR # This line ensures the DB is saved to disk
    )
    vectorstore.persist() # Explicitly save the vectorstore

    return vectorstore, split_docs

def setup_rag_chain(vectorstore):
    """
    Sets up the Retrieval-Augmented Generation (RAG) chain.
    This includes initializing the Groq LLM and combining it with the retriever.
    """
    if vectorstore is None:
        return None

    # Initialize the Groq Chat model for the main LLM interaction
    llm = ChatGroq(
        temperature=0.2, # Lower temperature for more factual, less creative responses for analysis
        model_name=GROQ_MODEL_NAME,
        groq_api_key=os.getenv("GROQ_API_KEY") # Ensure this env var is set
    )
    st.session_state.llm = llm # Store LLM instance in session state

    # Define the system prompt to guide the LLM's behavior as a resume analyzer
    system_prompt = (
        "You are an expert AI Resume Analyzer. Your primary task is to extract key information "
        "and provide insightful analysis about the uploaded resume based *only* on the provided context. "
        "Be factual, concise, and helpful. Do not invent information or make assumptions. "
        "If the information is not present in the resume, politely state that you cannot find the answer "
        "within the provided context. "
        "Focus on sections like: contact information, summary/objective, work experience, education, "
        "skills (technical and soft), projects, awards, and certifications. "
        "When asked for specific details, retrieve them directly and accurately."
        "\n\nContext from Resume:\n{context}"
    )

    # Create the chat prompt template for the LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Create a chain that combines the retrieved documents with the LLM's prompt
    Youtube_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create a retriever from the vectorstore to fetch relevant documents
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": DEFAULT_K_RETRIEVER})
    
    # Combine the retriever and the question-answer chain into the final RAG chain
    rag_chain = create_retrieval_chain(retriever, Youtube_chain)
    return rag_chain

# --- Streamlit UI Components ---
st.title("📄 ResumeGenie: Your AI Resume Analyzer")
st.markdown("Upload your resume (PDF only) and ask questions to get instant insights!")
st.info("💡 **Tip:** For best results, upload clear, text-searchable PDFs. Scanned image-based PDFs might not work well.")

# File Uploader Widget
uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"], key="resume_uploader")

# --- Document Processing Logic ---
# This block handles loading, processing, and setting up the RAG chain
# only when a new or different resume is uploaded.
if uploaded_file:
    # Calculate hash of the uploaded file's content to detect changes
    file_content_hash = calculate_file_hash(uploaded_file)
    
    # Check if a new file has been uploaded compared to the one in session state
    if st.session_state.current_file_hash != file_content_hash:
        st.session_state.current_file_hash = file_content_hash # Update the hash
        st.session_state.chat_history = [] # Clear chat history for the new resume
        st.session_state.vectorstore = None # Invalidate current vectorstore
        st.session_state.rag_chain = None # Invalidate current RAG chain

        uploaded_file_content = uploaded_file.getvalue()
        vectorstore, split_docs = process_resume(uploaded_file_content)

        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.rag_chain = setup_rag_chain(vectorstore)
            st.success(f"Successfully processed your resume into {len(split_docs)} chunks!")
            st.info("You can now ask questions about your resume using the chat below.")
        else:
            st.warning("Failed to process your resume. Please check the file and try again.")
            # Ensure state is reset if processing fails
            st.session_state.vectorstore = None
            st.session_state.rag_chain = None
            st.session_state.current_file_hash = None
            clear_chroma_db(CHROMA_DB_DIR) # Clear any partial DB

else:
    # If no file is currently uploaded (e.g., app just started or file removed),
    # ensure session state and persisted DB are clean.
    if st.session_state.current_file_hash is not None:
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.session_state.llm = None
        st.session_state.rag_chain = None
        st.session_state.current_file_hash = None
        clear_chroma_db(CHROMA_DB_DIR)
        st.info("Upload a PDF resume to begin analysis.")
    elif st.session_state.vectorstore is None: # Initial load state
        st.info("Please upload your resume (PDF) using the uploader above to start the analysis.")


# --- Chat Interface for Resume Analysis ---
if st.session_state.vectorstore: # Only show chat interface if a resume is loaded
    st.subheader("Ask About Your Resume")

    # Display chat history from session state
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "context" in message and message["context"]:
                with st.expander("🔍 View Context Used"):
                    for doc in message["context"]:
                        # Show a truncated preview of the page content from the retrieved document
                        st.markdown(f"- **Source:** {getattr(doc, 'metadata', {}).get('source', 'N/A')}"
                                    f" **Page:** {getattr(doc, 'metadata', {}).get('page', 'N/A')}\n"
                                    f"  ```\n{doc.page_content[:300]}...\n```") # Show a bit more context

    # --- Pre-defined Quick Analysis Prompts ---
    # Provide buttons for common resume analysis questions to guide students
    st.markdown("---")
    st.markdown("💡 **Quick Analysis Prompts:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Summarize my experience", key="btn_summary"):
            st.session_state.temp_query = "Provide a concise summary of my professional work experience and key achievements."
    with col2:
        if st.button("List my key skills", key="btn_skills"):
            st.session_state.temp_query = "What are my main technical and soft skills listed in the resume? List programming languages and tools."
    with col3:
        if st.button("Education details", key="btn_education"):
            st.session_state.temp_query = "Provide details about my education, including degrees, institutions, and dates."
    
    col4, col5 = st.columns(2)
    with col4:
        if st.button("What jobs did I have?", key="btn_jobs"):
            st.session_state.temp_query = "List all companies I have worked for, along with my job titles and dates of employment."
    with col5:
        if st.button("Any projects mentioned?", key="btn_projects"):
            st.session_state.temp_query = "Are there any personal projects or portfolio items mentioned in my resume? Describe them briefly."

    # --- Chat Input for User Queries ---
    # Determine the query from either the pre-defined buttons or the chat input box
    query = ""
    if "temp_query" in st.session_state and st.session_state.temp_query:
        query = st.session_state.temp_query
        del st.session_state.temp_query # Clear it after use
    else:
        query = st.chat_input("E.g., 'What are my responsibilities at [Company Name]?' or 'How many years of experience do I have?'")

    if query:
        # Add user's query to chat history for display
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        if st.session_state.rag_chain:
            with st.spinner("Analyzing your resume..."):
                try:
                    start_time = datetime.now()
                    response = st.session_state.rag_chain.invoke({"input": query})
                    end_time = datetime.now()
                    st.toast(f"Analysis completed in {(end_time - start_time).total_seconds():.2f} seconds.")

                    # Extract answer and context from the RAG chain's response
                    answer = response.get('answer', 'Sorry, I could not find that information in your resume based on the provided context.')
                    context = response.get('context', [])

                    # Add assistant's response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer, "context": context})

                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        if context:
                            with st.expander("🔍 View Context Used"):
                                for doc in context:
                                    st.markdown(f"- **Source:** {getattr(doc, 'metadata', {}).get('source', 'N/A')}"
                                                f" **Page:** {getattr(doc, 'metadata', {}).get('page', 'N/A')}\n"
                                                f"  ```\n{doc.page_content[:300]}...\n```")
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}. Please ensure your API keys are correct and try again.")
                    st.session_state.chat_history.append({"role": "assistant", "content": "An error occurred while analyzing your request. Please check API keys and try again."})
                    with st.chat_message("assistant"):
                        st.markdown("An error occurred while analyzing your request. Please try again.")
        else:
            st.warning("Please upload your resume first to start the analysis.")
            st.session_state.chat_history.append({"role": "assistant", "content": "No resume loaded. Please upload your resume to begin."})
            with st.chat_message("assistant"):
                st.markdown("No resume loaded. Please upload your resume to begin.")

st.markdown("---")
st.markdown("Developed with ❤️ by GraceUP Buddy for Students")



# ---- Simple Version ----

# import streamlit as st
# import os
# import shutil
# import asyncio
# from datetime import datetime
# import hashlib

# from langchain.docstore.document import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.prompts import ChatPromptTemplate
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import PyPDFLoader
# from dotenv import load_dotenv

# load_dotenv()

# RESUME_TEMP_DIR = "temp_resume_uploads"
# CHROMA_DB_DIR = "chroma_db_resume"
# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 200
# DEFAULT_K_RETRIEVER = 5
# GROQ_MODEL_NAME = "gemma2-9b-it"
# EMBEDDING_MODEL_NAME = "models/embedding-001"

# try:
#     asyncio.get_event_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())

# st.set_page_config(page_title="ResumeGenie: AI Resume Analyzer", layout="centered")

# if "vectorstore" not in st.session_state:
#     st.session_state.vectorstore = None
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "llm" not in st.session_state:
#     st.session_state.llm = None
# if "rag_chain" not in st.session_state:
#     st.session_state.rag_chain = None
# if "current_file_hash" not in st.session_state:
#     st.session_state.current_file_hash = None

# def clear_chroma_db(db_dir):
#     if os.path.exists(db_dir):
#         try:
#             if st.session_state.get("vectorstore"):
#                 try:
#                     st.session_state.vectorstore._client.close()
#                 except Exception:
#                     pass
#             st.cache_resource.clear()
#             shutil.rmtree(db_dir)
#             st.info(f"Cleared existing knowledge base at '{db_dir}'.")
#         except PermissionError:
#             st.warning("Chroma DB is currently in use. Try again shortly.")

# def calculate_file_hash(uploaded_file):
#     if uploaded_file is None:
#         return None
#     return hashlib.sha256(uploaded_file.getvalue()).hexdigest()

# @st.cache_resource(show_spinner="Analyzing resume...")
# def process_resume(uploaded_file_content):
#     if not uploaded_file_content:
#         return None, None
#     os.makedirs(RESUME_TEMP_DIR, exist_ok=True)
#     temp_pdf_path = os.path.join(RESUME_TEMP_DIR, f"resume_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.pdf")
#     try:
#         with open(temp_pdf_path, "wb") as f:
#             f.write(uploaded_file_content)
#         loader = PyPDFLoader(temp_pdf_path)
#         docs = loader.load()
#     except Exception as e:
#         st.error(f"Error loading PDF: {e}")
#         return None, None
#     finally:
#         if os.path.exists(temp_pdf_path):
#             os.remove(temp_pdf_path)
#         if not os.listdir(RESUME_TEMP_DIR):
#             os.rmdir(RESUME_TEMP_DIR)

#     if not docs:
#         st.warning("Could not extract readable text from resume.")
#         return None, None

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
#     split_docs = text_splitter.split_documents(docs)
#     clear_chroma_db(CHROMA_DB_DIR)
#     embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
#     vectorstore = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
#     vectorstore.persist()
#     return vectorstore, split_docs

# def setup_rag_chain(vectorstore):
#     if vectorstore is None:
#         return None

#     llm = ChatGroq(temperature=0.2, model_name=GROQ_MODEL_NAME, groq_api_key=os.getenv("GROQ_API_KEY"))
#     st.session_state.llm = llm
#     system_prompt = (
#         "You are an expert AI Resume Analyzer. Use the resume context provided to answer questions."
#         " If the information is not available, say so clearly."
#         "\n\nContext:\n{context}"
#     )
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ])
#     document_chain = create_stuff_documents_chain(llm, prompt)
#     retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": DEFAULT_K_RETRIEVER})
#     return create_retrieval_chain(retriever, document_chain)

# st.title("📄 ResumeGenie: Your AI Resume Analyzer")
# st.markdown("Upload your resume (PDF only) and ask questions!")

# uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# if uploaded_file:
#     file_hash = calculate_file_hash(uploaded_file)
#     if st.session_state.current_file_hash != file_hash:
#         st.session_state.current_file_hash = file_hash
#         st.session_state.chat_history = []
#         st.session_state.vectorstore = None
#         st.session_state.rag_chain = None
#         file_bytes = uploaded_file.getvalue()
#         vectorstore, split_docs = process_resume(file_bytes)
#         if vectorstore:
#             st.session_state.vectorstore = vectorstore
#             st.session_state.rag_chain = setup_rag_chain(vectorstore)
#             st.success(f"Resume processed into {len(split_docs)} chunks.")
#             st.info("Ask questions below.")
#         else:
#             st.warning("Could not process resume.")
#             clear_chroma_db(CHROMA_DB_DIR)

# if st.session_state.vectorstore:
#     st.subheader("Ask Questions")
#     for message in st.session_state.chat_history:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
#     query = st.chat_input("Ask something like 'Summarize my experience'")
#     if query:
#         st.session_state.chat_history.append({"role": "user", "content": query})
#         with st.chat_message("user"):
#             st.markdown(query)
#         if st.session_state.rag_chain:
#             with st.spinner("Analyzing..."):
#                 try:
#                     response = st.session_state.rag_chain.invoke({"input": query})
#                     answer = response.get("answer", "Could not find answer in resume.")
#                     context = response.get("context", [])
#                     st.session_state.chat_history.append({"role": "assistant", "content": answer, "context": context})
#                     with st.chat_message("assistant"):
#                         st.markdown(answer)
#                         if context:
#                             with st.expander("🔍 View Context"):
#                                 for doc in context:
#                                     st.markdown(f"```{doc.page_content[:300]}...```")
#                 except Exception as e:
#                     st.error(f"Error: {e}")
# else:
#     st.info("Please upload a resume PDF to get started.")
