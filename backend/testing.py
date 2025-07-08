import os
import tempfile
from typing import List, Tuple, Dict, Any, Generator

import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Configuration constants
EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
CHROMA_DB_PATH = "./resume_chroma_db"
COLLECTION_NAME = "resume_collection"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
DEFAULT_QUERY_RESULTS = 10
TOP_K_RERANK = 3
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# System prompt for the LLM
SYSTEM_PROMPT = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""


async def process_uploaded_document(uploaded_file: UploadedFile) -> List[Document]:
    """
    Processes an uploaded PDF file by converting it to text chunks.

    Takes an uploaded PDF file, saves it temporarily, loads and splits the content
    into text chunks using recursive character splitting.

    Args:
        uploaded_file: A Streamlit UploadedFile object containing the PDF file

    Returns:
        A list of Document objects containing the chunked text from the PDF

    Raises:
        IOError: If there are issues reading/writing the temporary file
        Exception: If PDF processing fails
    """
    temp_file_path = None
    try:
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as temp_file:
            temp_file.write(await uploaded_file.read())
            temp_file_path = temp_file.name

        # Load PDF document
        pdf_loader = PyMuPDFLoader(temp_file_path)
        document_pages = pdf_loader.load()

        # Configure text splitter with predefined parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )
        
        document_chunks = text_splitter.split_documents(document_pages)
        return document_chunks

    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        raise
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


def initialize_vector_collection() -> chromadb.Collection:
    """
    Gets or creates a ChromaDB collection for vector storage.

    Creates an Ollama embedding function using the configured model and initializes
    a persistent ChromaDB client. Returns a collection that can be used to store and
    query document embeddings.

    Returns:
        chromadb.Collection: A ChromaDB collection configured with the Ollama embedding
            function and cosine similarity space.
    
    Raises:
        Exception: If ChromaDB initialization fails
    """
    try:
        # Initialize Ollama embedding function
        embedding_function = OllamaEmbeddingFunction(
            url=OLLAMA_URL,
            model_name=EMBEDDING_MODEL,
        )

        # Create persistent ChromaDB client
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Get or create collection with specified configuration
        vector_collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        
        return vector_collection

    except Exception as e:
        st.error(f"Error initializing vector collection: {str(e)}")
        raise


def store_documents_in_vector_collection(document_chunks: List[Document], file_identifier: str) -> None:
    """
    Adds document splits to a vector collection for semantic search.

    Takes a list of document splits and adds them to a ChromaDB vector collection
    along with their metadata and unique IDs based on the filename.

    Args:
        document_chunks: List of Document objects containing text chunks and metadata
        file_identifier: String identifier used to generate unique IDs for the chunks

    Returns:
        None. Displays a success message via Streamlit when complete.

    Raises:
        Exception: If there are issues upserting documents to the collection
    """
    try:
        vector_collection = initialize_vector_collection()
        
        # Prepare data for batch insertion
        documents_content = []
        documents_metadata = []
        document_ids = []

        for chunk_index, document_chunk in enumerate(document_chunks):
            documents_content.append(document_chunk.page_content)
            documents_metadata.append(document_chunk.metadata)
            document_ids.append(f"{file_identifier}_{chunk_index}")

        # Batch upsert documents to collection
        vector_collection.upsert(
            documents=documents_content,
            metadatas=documents_metadata,
            ids=document_ids,
        )
        
        st.success(f"Successfully added {len(document_chunks)} document chunks to the vector store!")

    except Exception as e:
        st.error(f"Error storing documents in vector collection: {str(e)}")
        raise


def retrieve_relevant_documents(query_text: str, max_results: int = DEFAULT_QUERY_RESULTS) -> Dict[str, Any]:
    """
    Queries the vector collection with a given prompt to retrieve relevant documents.

    Args:
        query_text: The search query text to find relevant documents.
        max_results: Maximum number of results to return. Defaults to 10.

    Returns:
        dict: Query results containing documents, distances and metadata from the collection.

    Raises:
        Exception: If there are issues querying the collection.
    """
    try:
        vector_collection = initialize_vector_collection()
        query_results = vector_collection.query(
            query_texts=[query_text], 
            n_results=max_results
        )
        return query_results

    except Exception as e:
        st.error(f"Error retrieving relevant documents: {str(e)}")
        raise


def generate_llm_response(context_text: str, user_question: str) -> Generator[str, None, None]:
    """
    Calls the language model with context and prompt to generate a response.

    Uses Ollama to stream responses from a language model by providing context and a
    question prompt. The model uses a system prompt to format and ground its responses appropriately.

    Args:
        context_text: String containing the relevant context for answering the question
        user_question: String containing the user's question

    Yields:
        String chunks of the generated response as they become available from the model

    Raises:
        Exception: If there are issues communicating with the Ollama API
    """
    try:
        # Prepare messages for the LLM
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Context: {context_text}, Question: {user_question}",
            },
        ]

        # Stream response from Ollama
        response_stream = ollama.chat(
            model=LLM_MODEL,
            stream=True,
            messages=messages,
        )

        # Yield response chunks as they arrive
        for response_chunk in response_stream:
            if not response_chunk.get("done", True):
                content = response_chunk.get("message", {}).get("content", "")
                if content:
                    yield content
            else:
                break

    except Exception as e:
        st.error(f"Error generating LLM response: {str(e)}")
        raise


def rerank_documents_with_cross_encoder(query_text: str, document_list: List[str]) -> Tuple[str, List[int]]:
    """
    Re-ranks documents using a cross-encoder model for more accurate relevance scoring.

    Uses the MS MARCO MiniLM cross-encoder model to re-rank the input documents based on
    their relevance to the query prompt. Returns the concatenated text of the top 3 most
    relevant documents along with their indices.

    Args:
        query_text: The user's query text for ranking relevance
        document_list: List of document strings to be re-ranked.

    Returns:
        tuple: A tuple containing:
            - concatenated_relevant_text (str): Concatenated text from the top ranked documents
            - relevant_document_indices (list[int]): List of indices for the top ranked documents

    Raises:
        ValueError: If documents list is empty
        Exception: If cross-encoder model fails to load or rank documents
    """
    if not document_list:
        raise ValueError("Document list cannot be empty")

    try:
        # Initialize cross-encoder model
        cross_encoder_model = CrossEncoder(CROSS_ENCODER_MODEL)
        
        # Rank documents based on relevance to query
        ranking_results = cross_encoder_model.rank(
            query_text, 
            document_list, 
            top_k=TOP_K_RERANK
        )

        # Extract and concatenate top-ranked documents
        concatenated_relevant_text = ""
        relevant_document_indices = []

        for ranking_result in ranking_results:
            document_index = ranking_result["corpus_id"]
            concatenated_relevant_text += document_list[document_index]
            relevant_document_indices.append(document_index)

        return concatenated_relevant_text, relevant_document_indices

    except Exception as e:
        st.error(f"Error re-ranking documents: {str(e)}")
        raise


def normalize_filename(filename: str) -> str:
    """
    Normalizes a filename by replacing special characters with underscores.
    
    Args:
        filename: The original filename to normalize
        
    Returns:
        str: Normalized filename with special characters replaced
    """
    character_replacements = {"-": "_", ".": "_", " ": "_"}
    return filename.translate(str.maketrans(character_replacements))


def main():
    """Main application function that sets up the Streamlit interface."""
    
    # Configure Streamlit page
    st.set_page_config(page_title="CV Align", page_icon="🤖")
    
    # Sidebar for document upload and processing
    with st.sidebar:
        st.header("Resume Upload")
        
        uploaded_file = st.file_uploader(
            "**Upload Your Resume**", 
            type=["pdf"], 
            accept_multiple_files=False,
            help="Upload a PDF document to create a knowledge base for questions."
        )

        process_button = st.button(
            "Process CV",
            help="Process the uploaded PDF and add it to the vector store.",
            type="primary"
        )
        
        if uploaded_file and process_button:
            with st.spinner("Processing resume..."):
                try:
                    # Normalize filename for use as identifier
                    normalized_filename = normalize_filename(uploaded_file.name)
                    
                    # Process the uploaded document
                    document_chunks = process_uploaded_document(uploaded_file)
                    
                    # Store document chunks in vector collection
                    store_documents_in_vector_collection(document_chunks, normalized_filename)
                    
                except Exception as e:
                    st.error(f"Failed to process document: {str(e)}")

    # Main content area for questions and answers
    st.header("CV Align")
    st.markdown("Ask questions about your uploaded documents and get AI-powered answers.")
    
    user_question = st.text_area(
        "**Ask a question related to your document:**",
        height=100,
        placeholder="Enter your question here...",
        help="Type your question about the content in your uploaded PDF document."
    )
    
    ask_button = st.button(
        "Analyse",
        type="primary",
        help="Submit your question to get an AI-generated answer."
    )

    # Process question and generate answer
    if ask_button and user_question.strip():
        with st.spinner("Searching for relevant information..."):
            try:
                # Retrieve relevant documents from vector store
                search_results = retrieve_relevant_documents(user_question)
                retrieved_documents = search_results.get("documents", [[]])[0]
                
                if not retrieved_documents:
                    st.warning("No relevant documents found. Please upload and process a document first.")
                    return
                
                # Re-rank documents for better relevance
                relevant_context, relevant_indices = rerank_documents_with_cross_encoder(
                    user_question, retrieved_documents
                )
                
                # Generate and display AI response
                st.subheader("AI Response")
                llm_response = generate_llm_response(
                    context_text=relevant_context, 
                    user_question=user_question
                )
                st.write_stream(llm_response)

                # Display additional information in expandable sections
                with st.expander("📄 View Retrieved Resumes"):
                    st.json(search_results)

                with st.expander("🎯 Most Relevant Resume Details"):
                    st.write("**Relevant Resume Indices:**")
                    st.write(relevant_indices)
                    st.write("**Relevant Context:**")
                    st.text_area("Context", relevant_context, height=200)

            except Exception as e:
                st.error(f"An error occurred while processing your question: {str(e)}")
    
    elif ask_button and not user_question.strip():
        st.warning("Please enter a question before clicking 'Ask Question'.")


if __name__ == "__main__":
    main()