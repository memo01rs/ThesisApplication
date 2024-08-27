import streamlit as st
import logging
import os
import tempfile
import pdfplumber
import ollama

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from typing import List, Tuple, Dict, Any, Optional

# Streamlit page configuration
st.set_page_config(
    page_title="PDF File Analysis Hub",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("pdf_analysis_logger")


@st.cache_resource(show_spinner=True)
def retrieve_model_names(
    model_info: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    """
    Retrieves available model names from the model information.

    Args:
        model_info (Dict[str, List[Dict[str, Any]]]): Dictionary containing model details.

    Returns:
        List[str]: A list of model names.
    """
    logger.info("Retrieving model names from model information")
    model_names = [model["name"] for model in model_info["models"]]
    logger.info(f"Available models: {model_names}")
    return model_names


def build_vector_store(uploaded_pdf) -> Chroma:
    """
    Builds a vector database from an uploaded PDF document.

    Args:
        uploaded_pdf (st.UploadedFile): The uploaded PDF file.

    Returns:
        Chroma: A vector store containing the document chunks.
    """
    logger.info(f"Building vector store for file: {uploaded_pdf.name}")
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, uploaded_pdf.name)
        with open(file_path, "wb") as temp_file:
            temp_file.write(uploaded_pdf.getvalue())
        logger.info(f"File saved to temporary path: {file_path}")

        pdf_loader = UnstructuredPDFLoader(file_path)
        document_data = pdf_loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        document_chunks = splitter.split_documents(document_data)
        logger.info("Document split into chunks")

        embedding_model = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
        vector_store = Chroma.from_documents(
            documents=document_chunks, embedding=embedding_model, collection_name="pdf_qna"
        )
        logger.info("Vector store successfully created")

    return vector_store


def generate_answer(user_question: str, vector_store: Chroma, model_name: str) -> str:
    """
    Generates an answer to a user question using the vector store and selected model.

    Args:
        user_question (str): The user's question.
        vector_store (Chroma): The vector database containing document chunks.
        model_name (str): The name of the selected model.

    Returns:
        str: The generated response to the user's question.
    """
    logger.info(f"Processing question: {user_question} using model: {model_name}")

    llm = ChatOllama(model=model_name, temperature=0.2)
    retriever = vector_store.as_retriever()

    # System prompt instructing the AI to follow specific guidelines
    prompt_template = """
    You are an AI specialized in the analysis and processing of regulations.
    You must follow the user's instructions, responding in a detailed, precise, and contextually relevant manner.
    You must not 'hallucinate' or invent information that is not present in the document.

    Context:
    {context}

    Question: {question}

    Provide a detailed and precise answer using only the provided context.
    If the context does not contain the information, say 'I don't know'.
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    processing_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = processing_chain.invoke(user_question)
    logger.info("Question processed and response generated")
    return response


@st.cache_data
def convert_pdf_to_images(pdf_file) -> List[Any]:
    """
    Converts all pages of a PDF into images.

    Args:
        pdf_file (st.UploadedFile): The uploaded PDF file.

    Returns:
        List[Any]: A list of image objects representing each page of the PDF.
    """
    logger.info(f"Extracting pages as images from file: {pdf_file.name}")
    with pdfplumber.open(pdf_file) as pdf:
        pages_as_images = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages successfully extracted as images")
    return pages_as_images


def remove_vector_store(vector_store: Optional[Chroma]) -> None:
    """
    Removes the vector database and clears related session state.

    Args:
        vector_store (Optional[Chroma]): The vector database to be removed.
    """
    logger.info("Removing vector store")
    if vector_store:
        vector_store.delete_collection()
        for key in ["pdf_pages", "file_upload", "vector_store"]:
            st.session_state.pop(key, None)
        st.success("Collection and temporary files successfully deleted.")
        logger.info("Vector store and related session state cleared")
        st.rerun()
    else:
        st.error("No vector database found to delete.")
        logger.warning("No vector store found to delete")


def main_app() -> None:
    """
    Sets up the user interface, handles file uploads, processes user queries, and displays results.
    """
    st.subheader("📄 PDF File Analysis Hub", divider="gray", anchor=False)

    # Retrieve available models from Ollama
    model_info = ollama.list()
    available_models = retrieve_model_names(model_info)

    col1, col2 = st.columns([1.5, 2])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = None

    if available_models:
        selected_model = col2.selectbox(
            "Select a model available locally ↓", available_models
        )

    file_upload = col1.file_uploader(
        "Upload a PDF file ↓", type="pdf", accept_multiple_files=False
    )

    if file_upload:
        st.session_state["file_upload"] = file_upload
        if st.session_state["vector_store"] is None:
            st.session_state["vector_store"] = build_vector_store(file_upload)
        pdf_pages = convert_pdf_to_images(file_upload)
        st.session_state["pdf_pages"] = pdf_pages

        zoom_slider = col1.slider(
            "Zoom Level", min_value=100, max_value=1000, value=700, step=50
        )

        with col1:
            with st.container(height=500, border=True):
                for page_image in pdf_pages:
                    st.image(page_image, width=zoom_slider)

    delete_button = col1.button("⚠️  Delete collection", type="secondary")

    if delete_button:
        remove_vector_store(st.session_state["vector_store"])

    with col2:
        message_container = st.container(height=1000, border=True)

        for message in st.session_state["messages"]:
            avatar = "🤔" if message["role"] == "user" else "🧠"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if user_prompt := st.chat_input("Enter your question..."):
            try:
                st.session_state["messages"].append({"role": "user", "content": user_prompt})
                message_container.chat_message("user", avatar="🤔").markdown(user_prompt)

                with message_container.chat_message("assistant", avatar="🧠"):
                    with st.spinner(":green[Processing request...]"):
                        if st.session_state["vector_store"]:
                            response = generate_answer(
                                user_prompt, st.session_state["vector_store"], selected_model
                            )
                            st.markdown(response)
                        else:
                            st.warning("Please upload a PDF file first.")

                if st.session_state["vector_store"]:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(f"An error occurred: {e}", icon="⛔️")
                logger.error(f"Error processing the request: {e}")
        else:
            if st.session_state["vector_store"] is None:
                st.warning("Upload a PDF file to start the analysis...")


if __name__ == "__main__":
    main_app()

