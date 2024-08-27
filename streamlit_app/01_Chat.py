import ollama
import streamlit as st
from openai import OpenAI

# Page configuration with specific parameters
st.set_page_config(
    page_title="Regulatory AI Hub",  # Sets the title of the web page
    page_icon="️💬",  # Sets the icon of the web page
    layout="centered",  # Centers the layout of the page
    initial_sidebar_state="auto",  # Automatically manages the sidebar state
)


def get_available_model_names(models_metadata: dict) -> list:
    # Extracts the names of the available models from the provided metadata
    return [model["name"] for model in models_metadata.get("models", [])]


def construct_system_prompt(messages: list) -> list:
    # Constructs a system prompt by prepending an introductory system message
    intro_message = {
        "role": "system",  # Defines the role of the message as 'system'
        "content": (
            "You are an AI designed for in-depth analysis and interpretation of legal documents and regulations. "
            "Your task is to provide clear, concise, and precise responses tailored to the user's query, "
            "ensuring accuracy and context relevance."
        )
    }
    # Returns the list of messages with the system prompt at the beginning
    return [intro_message] + messages


def run_app():
    # Main function to run the Streamlit application
    st.title("Legal AI Assistant")  # Sets the title of the application

    # Initializes the OpenAI client with a custom base URL and API key
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    # Retrieves available models metadata and extracts model names
    models_metadata = ollama.list()
    model_options = get_available_model_names(models_metadata)

    if model_options:
        # Displays a radio button for model selection if models are available
        selected_model = st.radio("Choose a locally available model:", model_options)
    else:
        # Shows a warning if no models are available and provides a settings button
        st.warning("No models are available locally. Please visit the settings to download one.", icon="⚠️")
        if st.button("Go to Settings"):
            st.page_switch("pages/03_⚙️_Settings.py")  # Switches to the settings page

    # Creates a container for displaying the conversation history
    conversation_container = st.container()

    # Initializes the session state for conversation history if not already present
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Loops through the conversation history and displays each message
    for msg in st.session_state.conversation_history:
        avatar = "⚖️" if msg["role"] == "assistant" else "👤"  # Chooses an avatar based on the message role
        with conversation_container.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])  # Displays the message content

    # Handles user input through the chat input box
    if user_input := st.chat_input("Enter your query..."):
        try:
            # Appends the user's message to the conversation history
            st.session_state.conversation_history.append({"role": "user", "content": user_input})

            # Displays the user's message in the conversation container
            conversation_container.chat_message("user", avatar="👤").markdown(user_input)

            with conversation_container.chat_message("assistant", avatar="⚖️"):
                with st.spinner("Processing request..."):
                    # Sends the conversation history to the selected model for a response
                    stream = client.chat.completions.create(
                        model=selected_model,
                        messages=construct_system_prompt(
                            [{"role": m["role"], "content": m["content"]} for m in
                             st.session_state.conversation_history]
                        ),
                        stream=True,
                    )
                response_text = st.write_stream(stream)  # Writes the streaming response to the UI

            # Appends the assistant's response to the conversation history
            st.session_state.conversation_history.append({"role": "assistant", "content": response_text})

        except Exception as err:
            # Displays an error message if an exception occurs
            st.error(f"An error occurred: {err}", icon="❗")


if __name__ == "__main__":
    run_app()  # Executes the application if the script is run directly
