import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA

# Load env variables
load_dotenv()

def query_data(question):
    """
    Query the policy document database with a natural language question.
    
    This function connects to a Pinecone vector store, retrieves relevant 
    document chunks based on the question, and uses Google's Gemini model 
    to generate a contextual answer.
    
    Args:
        question (str): The natural language question to ask about the policies
        
    Returns:
        None: Prints the answer to stdout
        
    Example:
        >>> query_data("What is the policy on equipment?")
        --- Asking: What is the policy on equipment? ---
        
        Answer: 
        [Generated answer based on policy documents]
    """
    print(f"\n--- Asking: {question} ---")

    # Initalize embedding model (same model as ingestion)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Connect to Pinecone Index
    vectorstore = PineconeVectorStore(
        index_name=os.getenv('PINECONE_INDEX_NAME'),
        embedding=embeddings
    )

    # Set up LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # Create retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vectorstore.as_retriever()
    )

    # Run the query
    response = qa_chain.invoke({"query": question})

    print("\nAnswer: ")
    print(response["result"])

if __name__ == "__main__":
    # Testing queries
    query_data("What is the policy on equipment?")
    query_data("Can I work from a cafe?")