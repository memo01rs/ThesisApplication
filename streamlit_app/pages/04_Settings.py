import streamlit as st
import ollama
from time import sleep

# Configure the Streamlit page
st.set_page_config(
    page_title="Manage AI Models",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)


def model_management_app():
    """
    Main function to manage the UI for downloading and deleting AI models.
    Users can interactively download or remove models.
    """
    st.subheader("AI Model Management", divider="blue", anchor=False)

    # Section to download new models
    st.subheader("Download New Models", anchor=False)
    model_input = st.text_input(
        "Enter the name of the model to download ↓", placeholder="e.g., llama2"
    )

    if st.button(f"📥 :blue[**Download**] :orange[{model_input}]"):
        if model_input:
            try:
                # Try to download the specified model
                ollama.pull(model_input)
                st.success(f"Successfully downloaded model: {model_input}", icon="🎉")
                st.balloons()
                sleep(1)  # Brief pause for user experience enhancement
                st.experimental_rerun()  # Refresh the interface to update changes
            except Exception as error:
                # Handle errors during the download process
                st.error(
                    f"Unable to download model: {model_input}. Error: {str(error)}",
                    icon="😵",
                )
        else:
            st.warning("Please provide a model name before attempting to download.", icon="⚠️")

    st.divider()

    # Section to delete existing models
    st.subheader("Remove Existing Models", anchor=False)
    models_details = ollama.list()
    models_available = [model["name"] for model in models_details["models"]]

    if models_available:
        # Allow the user to select models for deletion
        models_to_remove = st.multiselect("Choose models to delete", models_available)
        if st.button("🗑️ :red[**Delete Selected Models**]"):
            for model_name in models_to_remove:
                try:
                    # Attempt to delete the selected model
                    ollama.delete(model_name)
                    st.success(f"Model deleted: {model_name}", icon="✅")
                    st.balloons()
                    sleep(1)  # Brief pause for user experience enhancement
                    st.experimental_rerun()  # Refresh the interface to update changes
                except Exception as error:
                    # Handle errors during the deletion process
                    st.error(
                        f"Unable to delete model: {model_name}. Error: {str(error)}",
                        icon="😳",
                    )
    else:
        # Notify the user if no models are available for deletion
        st.info("No models currently available for deletion.", icon="🦗")


if __name__ == "__main__":
    model_management_app()
