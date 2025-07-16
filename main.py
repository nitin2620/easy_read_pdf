__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import tempfile
from dotenv import load_dotenv
import os
import os
from groq import Groq
from chromadb.config import Settings


# Load .env file
load_dotenv()

# Access the keys
key = st.secrets["key"]






# Session state to remember the uploaded file and extracted text
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = ""
if 'uploaded' not in st.session_state:
    st.session_state.uploaded = False
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = None
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'splitter' not in st.session_state:
    st.session_state.splitter = None

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    loader = PyPDFLoader(uploaded_file)
    documents = loader.load()
    return documents

# Page 1: Upload PDF
if not st.session_state.uploaded:
    st.title("📄 Upload a PDF")
    os.environ["STREAMLIT_HOME"] = "/tmp"
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    
    st.chroma_client = chromadb.Client()
    try:
        st.collection = st.chroma_client.delete_collection(name="my_collection")
    except:
        pass
    st.collection = st.chroma_client.create_collection(name="my_collection")
    st.splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )


    if uploaded_file is not None:
        with st.spinner("Reading PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            st.session_state.pdf_text = extract_text_from_pdf(tmp_file_path)
            chunks = st.splitter.split_documents(st.session_state.pdf_text)
            
            chunks=[chunk.page_content for chunk in chunks]
            chunks=[ str(chunk) for chunk in chunks]
            ids=list(range(0,len(chunks)))
            ids=[ str(id )for id in ids]
            st.collection.add(
                ids,
                documents=chunks
            )
            st.session_state.uploaded = True
            st.success("PDF uploaded and processed!")
        st.rerun()

# Page 2: Ask questions
else:
    st.title("💬 Ask Questions About the PDF")

    question = st.text_input("Enter your question:")
    
    if st.button("Get Answer") and question:
        # Very basic "answer" logic (just showing a matching sentence)
        results = st.collection.query(
            query_texts=[question], # Chroma will embed this for you
            n_results=2 # how many results to return
        )
        prompt=f'''
        <system>

        **task**:your task is provide the answer to user query by referencing the document provided below.

        **documents**
        document1:{results["documents"][0][0]}
        document2:{results["documents"][0][1]}

        **guidelines**
        -response should be short and confident

        **query**
        query: {question}


        **answer**





        </system>
        '''
        # key = os.getenv("key")
        client = Groq(
            api_key=key,
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-8b-instant",
        )

        st.write(chat_completion.choices[0].message.content)
                

    st.button("🔄 Upload New PDF", on_click=lambda: st.session_state.update(uploaded=False, pdf_text=""))

