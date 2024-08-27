import streamlit as st
import logging
import ollama
import chardet  # For detecting the encoding of text files

from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from typing import List, Tuple, Dict, Any, Optional

# Configure the Streamlit page
st.set_page_config(
    page_title="TXT File Analysis Hub",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("text_analysis_app")


@st.cache_resource(show_spinner=True)
def get_model_names(models_info: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """
    Retrieve the names of available models.

    Args:
        models_info (Dict[str, List[Dict[str, Any]]]): Dictionary with information about models.

    Returns:
        List[str]: List of model names.
    """
    logger.info("Retrieving model names from models_info")
    model_names = [model["name"] for model in models_info["models"]]
    logger.info(f"Available models: {model_names}")
    return model_names


def load_text_file(file) -> str:
    file_bytes = file.read()
    detected_encoding = chardet.detect(file_bytes)['encoding']
    return file_bytes.decode(detected_encoding)


def build_vector_database(uploaded_file) -> Chroma:
    logger.info(f"Building vector database from file: {uploaded_file.name}")

    # Read the content of the uploaded text file
    file_content = load_text_file(uploaded_file)

    # Split the document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=180)
    chunks = splitter.split_text(file_content)
    logger.info("Document has been split into chunks")

    # Generate embeddings and create the vector database
    embedding_model = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    vector_store = Chroma.from_texts(
        texts=chunks, embedding=embedding_model, collection_name="text_analysis"
    )
    logger.info("Vector database created")

    return vector_store


def remove_vector_database(vector_store: Optional[Chroma]) -> None:
    """
    Removes the vector database and clears session state.

    Args:
        vector_store (Optional[Chroma]): Vector store to be removed.
    """
    logger.info("Removing vector database")
    if vector_store is not None:
        vector_store.delete_collection()
        st.session_state.pop("uploaded_file", None)
        st.session_state.pop("vector_store", None)
        st.success("Collection and temporary files have been deleted.")
        logger.info("Vector database and session state cleared")
        st.rerun()
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector database, but none was found")


def generate_response(question: str, vector_store: Chroma, model_name: str) -> str:
    """
    Generates a response to a user's question using the vector store and selected model.

    Args:
        question (str): The user's question.
        vector_store (Chroma): The vector database containing document embeddings.
        model_name (str): The name of the selected model.

    Returns:
        str: The generated response.
    """
    logger.info(f"Processing question: {question} using model: {model_name}")

    # Initialize the language model and retriever
    language_model = ChatOllama(model=model_name, temperature=0.2)
    retriever = vector_store.as_retriever()

    # Create the chat prompt template with the specified instructions
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI specialized in the analysis and processing of regulations. "
                "Your task is to process the document uploaded by the user and answer the questions accurately and in detail. "
                "You must not 'hallucinate' or invent information that is not present in the document. "
                "If you cannot answer the question based on the document, say 'I don't know' instead of making up an answer.",
            ),
            (
                "user",
                "Context:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )

    # Combine the prompt with the model
    processing_chain = prompt | language_model

    # Retrieve relevant context from the vector store
    relevant_contexts = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in relevant_contexts])

    # Invoke the chain with the specific input values
    response = processing_chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    # Since response is likely an AIMessage object, directly access the content attribute
    clean_response = response.content

    logger.info("Question processed and response generated")
    return clean_response


def main_app() -> None:
    """
    Main function to set up and run the Streamlit application.
    """
    st.subheader("📄 TXT File Analysis Hub", divider="gray", anchor=False)

    # Retrieve available models from Ollama
    models_info = ollama.list()
    available_models = get_model_names(models_info)

    # Initialize session state for messages and vector store
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = None

    # User selects a model from available models
    if available_models:
        selected_model = st.selectbox(
            "Select a locally available model ↓", available_models
        )

    # Handle file upload for text files
    uploaded_file = st.file_uploader(
        "Upload a text file ↓", type=["txt"], accept_multiple_files=False
    )

    if uploaded_file:
        st.session_state["uploaded_file"] = uploaded_file
        if st.session_state["vector_store"] is None:
            st.session_state["vector_store"] = build_vector_database(uploaded_file)

    # Button to delete the vector database
    if st.button("⚠️  Delete collection"):
        remove_vector_database(st.session_state["vector_store"])

    # Display conversation history and handle user input
    message_container = st.container()

    for message in st.session_state["messages"]:
        avatar = "🧠" if message["role"] == "assistant" else "🤔"
        with message_container.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if user_input := st.chat_input("Enter your question here..."):
        try:
            st.session_state["messages"].append({"role": "user", "content": user_input})
            message_container.chat_message("user", avatar="🤔").markdown(user_input)

            with message_container.chat_message("assistant", avatar="🧠"):
                with st.spinner(":green[Generating response...]"):
                    if st.session_state["vector_store"] is not None:
                        response = generate_response(
                            user_input, st.session_state["vector_store"], selected_model
                        )
                        st.markdown(response)
                    else:
                        st.warning("Please upload a text file first.")

            if st.session_state["vector_store"] is not None:
                st.session_state["messages"].append(
                    {"role": "assistant", "content": response}
                )

        except Exception as e:
            st.error(f"Error: {e}", icon="⛔️")
            logger.error(f"Error during processing: {e}")
    else:
        if st.session_state["vector_store"] is None:
            st.warning("Please upload a text file to begin...")


if __name__ == "__main__":
    main_app()
