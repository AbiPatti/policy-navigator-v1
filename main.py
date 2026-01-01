import os
import uuid
import chainlit as cl
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Load env variables
load_dotenv()

# Load embeddings globally to prevent freezing on every message
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
print("Model loaded.")

@cl.on_chat_start
async def start():
    """
    Initialize the chat session and prompt user to upload a file.
    No database is loaded until the user uploads a file.
    """
    # Generate a unique namespace for this user session
    session_namespace = f"user_session_{uuid.uuid4().hex[:12]}"
    cl.user_session.set("namespace", session_namespace)
    
    # Prompt user to upload a file
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Upload a policy document to begin.",
            accept=["text/plain", "application/pdf"],
            max_size_mb=10,
            timeout=180
        ).send()
    
    # Get the uploaded file
    file = files[0]
    
    # Show processing message
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()
    
    # Ingest the uploaded file into Pinecone with namespace
    await ingest_file(file, session_namespace, msg)

async def ingest_file(file: cl.File, namespace: str, msg: cl.Message):
    """
    Ingest the uploaded file into Pinecone vector store with a unique namespace.
    
    Args:
        file: The uploaded Chainlit file object
        namespace: Unique namespace for this user session
        msg: Message object to update with progress
    """
    try:
        # Load file based on file extension
        if file.name.endswith('.pdf'):
            loader = PyPDFLoader(file.path)
        else:
            loader = TextLoader(file.path, encoding='utf-8')
        
        raw_documents = loader.load()
        print(f"DEBUG: Loaded {len(raw_documents)} raw documents")
        
        # Check if file was actually read
        if len(raw_documents) == 0:
            msg.content = f"Error: Could not read text from `{file.name}`. Is it an image-based PDF?"
            await msg.update()
            return
        
        # Check content length
        total_chars = sum(len(doc.page_content) for doc in raw_documents)
        print(f"DEBUG: Total characters: {total_chars}")
        
        if total_chars == 0:
            msg.content = f"Error: PDF appears to be empty or image-based. No text extracted from `{file.name}`."
            await msg.update()
            return
        
        # Print first 200 chars to debug
        if raw_documents:
            print(f"DEBUG: First 200 chars: {raw_documents[0].page_content[:200]}")
        
        # Split into chunks with more lenient settings
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False
        )
        documents = text_splitter.split_documents(raw_documents)
        print(f"DEBUG: Created {len(documents)} chunks")
        
        # If splitting fails, use raw documents directly
        if len(documents) == 0:
            print("DEBUG: Splitting returned 0 chunks, using raw documents")
            documents = raw_documents
        
        msg.content = f"Found {len(documents)} chunks. Uploading to Pinecone..."
        await msg.update()
        
        # Upload to Pinecone - do it synchronously to catch errors
        print(f"DEBUG: Uploading to namespace {namespace}")
        vectorstore = await cl.make_async(PineconeVectorStore.from_documents)(
            documents=documents,
            embedding=embeddings,
            index_name=os.getenv("PINECONE_INDEX_NAME"),
            namespace=namespace
        )
        print(f"DEBUG: Upload complete")
        
        msg.content = f"Uploaded {len(documents)} chunks. Ask questions about `{file.name}`."
        await msg.update()
        
    except Exception as e:
        print(f"ERROR in ingest_file: {str(e)}")
        import traceback
        traceback.print_exc()
        msg.content = f"Error processing file: {str(e)}"
        await msg.update()

@cl.on_message
async def main(message: cl.Message):
    """
    Handle user questions and query the vector store using the session's namespace.
    """
    # Get the session namespace
    namespace = cl.user_session.get("namespace")
    
    # Connect to Pinecone with the specific namespace
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings,
        namespace=namespace
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    # Create the retrieval chain with custom prompt
    from langchain_classic.prompts import PromptTemplate
    
    prompt_template = """Use the following context to answer the question. Keep your answer simple, casual, and conversational. Use everyday language like you're explaining to a friend. No formal or corporate speak.

Context: {context}

Question: {question}

Answer in a casual, simple way:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    # Run query
    res = await qa_chain.ainvoke({"query": message.content})
    
    # Check if relevant documents were found
    if not res.get("source_documents"):
        await cl.Message(content="No relevant information found in the document for that question.").send()
    else:
        await cl.Message(content=res["result"]).send()