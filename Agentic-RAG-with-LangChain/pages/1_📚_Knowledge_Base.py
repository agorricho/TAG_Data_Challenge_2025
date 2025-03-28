### IMPORT DEPENDENCIES ###

# import basics
import os
import glob
from dotenv import load_dotenv
import time
import tempfile
import shutil  # Add this import for removing directories

# import streamlit
import streamlit as st

# import langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings


# import supabase
from supabase.client import Client, create_client

### LOAD ENVIRONMENT VARIABLES - SETUP CONTEXT VARIABLES ###

# load environment variables
load_dotenv()  

# initiate supabase db
supabase_url = os.environ.get("SUPABASE_URL")
print(supabase_url)
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
print(supabase_key)
supabase: Client = create_client(supabase_url, supabase_key)

# initiate embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# Get the directory where the Streamlit app is running
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the persistent storage folder inside the app directory
UPLOAD_DIR = os.path.join(APP_DIR, "documents")

# Ensure the directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Define persistent storage directory for URLs
URL_DIR = os.path.join(APP_DIR, "URLs")

# Ensure the directory exists
os.makedirs(URL_DIR, exist_ok=True)

# Store URLs in a persistent file
url_file_path = os.path.join(URL_DIR, "articles.txt")

# Get the list of existing files in /documents
existing_files = set(os.listdir(UPLOAD_DIR))


### SET UP STREAMLIT APP ###

# initiating streamlit app
st.set_page_config(page_title="Agentic RAG Chatbot", 
                   page_icon="🤖",
                   layout="wide")

# Add a sidebar
with st.sidebar:
    st.title("Admin Controls")
    
    # Add deletion functionality section
    st.subheader("Delete Knowledge Base")
    
    # Add confirmation checkboxes for safety
    confirm_docs = st.checkbox("I want to delete all documents", key="confirm_docs")
    confirm_urls = st.checkbox("I want to delete all URLs", key="confirm_urls")
    
    # Add delete button that checks for confirmation
    if st.button("Delete Selected Data", key="delete_button"):
        if confirm_docs or confirm_urls:
            if confirm_docs:
                # Delete all files in the documents folder
                for file_path in glob.glob(os.path.join(UPLOAD_DIR, "*")):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        st.error(f"Error deleting {file_path}: {e}")
                st.success("All documents have been deleted!")
                # Reset the existing_files set
                existing_files = set()
            
            if confirm_urls:
                # Delete the URLs file
                if os.path.exists(url_file_path):
                    try:
                        os.remove(url_file_path)
                        # Create an empty file to maintain structure
                        with open(url_file_path, "w") as f:
                            pass
                        st.success("All URLs have been deleted!")
                    except Exception as e:
                        st.error(f"Error deleting URLs file: {e}")
            
            # Add a button to clear the success messages
            if st.button("Clear Messages", key="clear_messages"):
                st.experimental_rerun()
        else:
            st.warning("Please confirm which data you want to delete by checking the boxes above.")


### SET UP FRONT AND BACK END ###

# layout for main page
col_11, col_12 = st.columns([1, 5])

with col_11:
   st.image('inline-logo-with-tagline-and-path-2048x1152.png')

with col_12:
   st.header('Knowledge Base')

# layout for page fuinctionality
col_21, col_22, col_23 = st.columns([5,1,5])

# Load PDF documents
with col_21:
    st.markdown('### Upload Documents')
    pdf_docs = st.file_uploader('Upload documents and click on "Send & Process"', 
                                accept_multiple_files=True, 
                                key="pdf_uploader")
    
    # Add a placeholder to display processing messages
    message_placeholder = st.empty()

    # cereate a filed upload widget
    if st.button("Send & Process", 
                key="process_button"):
        
        # use spinner widget
        with st.spinner("Processing..."):
            for doc in pdf_docs:
                
                # Define the file path in the persistent /documents folder
                file_path = os.path.join(UPLOAD_DIR, doc.name)

                # Check if the file already exists
                if doc.name in existing_files:
                    message_placeholder.write(f'File "{doc.name}" already exists.\n\n Skipping upload!')
                    time.sleep(3)

                    # clear screen
                    message_placeholder.empty()
                    
                    # Skip to the next file
                    continue
                
                # echo messsage to screen 
                message_placeholder.write(f'Loading file: "{doc.name}"...')

                # Define the file path in the persistent /documents folder
                file_path = os.path.join(UPLOAD_DIR, doc.name)

                # Save uploaded file to persistent storage
                with open(file_path, "wb") as f:
                    f.write(doc.getbuffer())

                # Instantiate document loader with the persistent file path
                loader = PyPDFLoader(file_path)

                # split the documents in multiple chunks
                pdf_pages = loader.load()

                # echo messsage to screen 
                message_placeholder.write(f'File {doc.name} has {len(pdf_pages)} pages')

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_documents(pdf_pages)

                # echo messsage to screen 
                message_placeholder.write(f'File {doc.name} has {len(chunks)} chunks')

                # echo messsage to screen 
                message_placeholder.write(f'Uploading {doc.name} to vector store...')
                
                # store chunks in vector store
                vector_store = SupabaseVectorStore.from_documents(
                    chunks,
                    embeddings,
                    client=supabase,
                    table_name="documents",
                    query_name="match_documents",
                    chunk_size=1000,
                )

                # echo messsage to screen 
                message_placeholder.write(f'File {doc.name} uploaded to vector store')

                # wait enough time for user to see message
                time.sleep(3)

                # clear screen
                message_placeholder.empty()

    # Display available documents in the local file repository
    st.markdown("### Documents in the Vector Store:")

    # List all files in the /documents folder
    all_files = os.listdir(UPLOAD_DIR)
    
    # For document checkboxes
    selected_files = [file for idx, file in enumerate(all_files) if st.checkbox(file, value=True, key=f"doc_{idx}")]

# empty placeholder
with col_22:
    st.write("")

# load web articles
with col_23:
    st.markdown("### Upload Articles")

    # Create a placeholder for the text input
    url_input_placeholder = st.empty()

    # Text input box for entering article URLs
    with url_input_placeholder:
        article_urls = st.text_area("Enter article URLs (one per line)", key="url_input")

    # Read existing URLs from the stored file
    existing_urls = set()
    if os.path.exists(url_file_path):
        with open(url_file_path, "r") as f:
            existing_urls = set(line.strip() for line in f.readlines())

    # Button to process URLs
    if st.button("Send & Process Articles", key="process_articles_button"):
        
        # Split URLs into a list and remove empty lines
        url_list = [url.strip() for url in article_urls.split("\n") if url.strip()]
        
        # Filter out already uploaded URLs
        new_urls = [url for url in url_list if url not in existing_urls]

        if not new_urls:
            st.warning("All entered URLs have already been uploaded.")
        else:
            with open(url_file_path, "a") as f:
                for url in new_urls:
                    f.write(url + "\n")
            
            # Clear input field
            url_input_placeholder = st.empty()

            # Process each new article using LangChain WebBaseLoader
            with st.spinner("Processing articles..."):
                for url in new_urls:
                    loader = WebBaseLoader(url)
                    article_docs = loader.load()

                    # Split documents (if needed)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    chunks = text_splitter.split_documents(article_docs)

                    # Store chunks in vector store
                    vector_store = SupabaseVectorStore.from_documents(
                        chunks,
                        embeddings,
                        client=supabase,
                        table_name="documents",
                        query_name="match_articles",
                        chunk_size=1000,
                    )

    # Display available articles in the vector store
    st.markdown("### Articles in the Vector Store:")

    # Read and display articles from the stored file
    if os.path.exists(url_file_path):
        with open(url_file_path, "r") as f:
            all_articles = f.readlines()

        # For article checkboxes
        selected_articles = [article.strip() for idx, article in enumerate(all_articles) if st.checkbox(article.strip(), value=True, key=f"article_{idx}")]