import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

def ingest_docs():
    """
    Loads text file, splits into chunks, and uploads to Pinecone vector store.
    """
    print("Starting ingestion...")

    # Initialize embedding model (runs locally, no API needed)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Load text file
    print("Loading policy...")
    loader = TextLoader("data/policy.txt")
    raw_documents = loader.load()
    print("Loaded file")

    # Split into smaller chunks for better retrieval
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Characters per chunk
        chunk_overlap=50  # Overlap to preserve context
    )
    documents = text_splitter.split_documents(raw_documents)
    print(f"Created {len(documents)} chunks")

    # Upload to Pinecone vector database
    print("Connecting to Pinecone...")
    print("Uploading...")
    
    PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=os.getenv("PINECONE_INDEX_NAME")
    )
    
    print("Ingestion complete")

if __name__ == "__main__":
    ingest_docs()