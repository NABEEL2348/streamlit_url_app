import os
import time
import nltk  # Add this import
from dotenv import load_dotenv
import streamlit as st

from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import ChatCohere
from langchain.chains import RetrievalQAWithSourcesChain

# Load environment variables
load_dotenv()

# Download necessary NLTK resources
nltk.download('punkt')

# Ensure COHERE_API_KEY is loaded from .env
cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    st.error("⚠️ COHERE_API_KEY not found in environment variables.")
    st.stop()

# Initialize embeddings and LLM
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=cohere_api_key
)

llm = ChatCohere(cohere_api_key=cohere_api_key)

# Streamlit UI setup
st.title("📰 News Research Tool")
st.sidebar.title("🔗 News Article URLs")

# Collect URLs from the sidebar
urls = [st.sidebar.text_input(f"Enter URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("🚀 Process URLs")
mainplaceholder = st.empty()

# Process URLs if button is clicked
if process_url_clicked:
    if not any(urls):  # Ensure at least one URL is provided
        st.error("⚠️ Please enter at least one URL.")
    else:
        try:
            mainplaceholder.text("🔄 Loading data from URLs...")

            # Load and process documents from URLs
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()

            st.write(f"✅ Loaded {len(data)} documents.")

            # Split the documents into chunks
            splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            docs = splitter.split_documents(data)
            st.write(f"🔍 Split into {len(docs)} chunks.")

            if not docs:
                st.error("⚠️ No valid content found to embed. Please check the URLs.")
            else:
                # Create embeddings and vector index
                mainplaceholder.text("🔧 Creating embeddings and vector index...")
                vector_index = FAISS.from_documents(docs, embeddings)
                vector_index.save_local("faiss_index_cohere")
                mainplaceholder.success("✅ Vector index created and saved successfully!")

        except Exception as e:
            st.error(f"❌ An error occurred while processing URLs: {str(e)}")

# Query the knowledge base
query = st.text_input("Ask a question about the articles:")

if query:
    if os.path.exists("faiss_index_cohere"):
        try:
            # Load the vector store
            vectorstore = FAISS.load_local("faiss_index_cohere", embeddings, allow_dangerous_deserialization=True)

            # Set up retriever and QA chain
            retriever = vectorstore.as_retriever()
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

            # Get the answer
            result = chain({"question": query}, return_only_outputs=True)

            # Display the result
            st.header("📌 Answer")
            st.subheader(result["answer"])
            st.markdown("📚 **Sources:**")
            st.code(result["sources"])

        except Exception as e:
            st.error(f"❌ Error while processing your question: {str(e)}")
    else:
        st.warning("⚠️ Please process URLs first to generate a vector index.")
