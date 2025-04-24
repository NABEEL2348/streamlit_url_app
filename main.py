import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
cohere_api_key="2aIcYOdp7Ah1CUT0esEGC6I24MBXpcPosrW3p2ui"
if not cohere_api_key:
    st.error("⚠️ COHERE_API_KEY not found in environment variables.")
    st.stop()

# Set up embeddings and LLM
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=cohere_api_key
)
llm = ChatCohere(cohere_api_key=cohere_api_key)

# Streamlit UI
st.title("📰 News Research Tool")
st.sidebar.title("🔗 News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"Enter URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("🚀 Process URLs")
mainplaceholder = st.empty()

# Load and embed documents
if process_url_clicked:
    if not urls:
        st.error("⚠️ Please enter at least one URL.")
    else:
        try:
            mainplaceholder.text("🔄 Loading content from web pages...")

            loader = WebBaseLoader(urls)
            data = loader.load()
            st.write(f"✅ Loaded {len(data)} documents.")

            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", ","],
                chunk_size=1000,
                chunk_overlap=100
            )
            docs = splitter.split_documents(data)
            st.write(f"🔍 Split into {len(docs)} text chunks.")

            if not docs:
                st.error("⚠️ No valid content found to embed. Please check the URLs.")
            else:
                mainplaceholder.text("🔧 Creating vector index using Cohere embeddings...")

                vector_index = FAISS.from_documents(docs, embeddings)
                vector_index.save_local("faiss_index_cohere")

                mainplaceholder.success("✅ Vector index created and saved successfully!")

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

# QA section
query = st.text_input("Ask a question about the articles:")

if query:
    if os.path.exists("faiss_index_cohere"):
        try:
            vectorstore = FAISS.load_local("faiss_index_cohere", embeddings, allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever()
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

            result = chain({"question": query}, return_only_outputs=True)

            st.header("📌 Answer")
            st.subheader(result["answer"])
            st.markdown("📚 **Sources:**")
            st.code(result["sources"])

        except Exception as e:
            st.error(f"❌ Error while processing your question: {str(e)}")
    else:
        st.warning("⚠️ Please process URLs first to generate a knowledge base.")
