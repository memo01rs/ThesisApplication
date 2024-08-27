import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl
from langchain_core.prompts import ChatPromptTemplate

# Define the prompt template for the LLM
template = """
You are an AI assistant.
Answer the following question with only three sentences and nothing more!

Context: {context}

{file_contents}

Question: {question}

Antwort:
"""

prompt = ChatPromptTemplate.from_template(template)  # Create the prompt template


@cl.on_chat_start
async def on_chat_start():
    """
    Handles the chat session start event.
    Initializes the conversation by asking the user to upload a PDF file,
    processes the file, and sets up the retrieval chain for answering questions.
    """
    global file_contents
    file_contents = ""  # Initialize file_contents as an empty string

    # Wait for the user to upload a PDF file
    files = await cl.AskFileMessage(
        content="Please upload a PDF file to begin!",
        accept=["application/pdf"],
        max_size_mb=100,  # Optionally limit the file size
        timeout=180,  # Set a timeout for user response
    ).send()

    file = files[0]  # Get the first uploaded file

    # Inform the user that processing has started
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Read the PDF file and extract text
    pdf = PyPDF2.PdfReader(file.path)
    for page in pdf.pages:
        file_contents += page.extract_text()  # Append each page's text to file_contents

    # Split the extracted text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=180)
    texts = text_splitter.split_text(file_contents)

    # Create metadata for each text chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store for document retrieval
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    # Initialize message history for conversation context
    message_history = ChatMessageHistory()

    # Set up memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a retrieval chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        ChatOllama(model="mistral-nemo"),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Store the chain and prompt template in the user session
    cl.user_session.set("chain", chain)
    cl.user_session.set("prompt_template", prompt)

    # Inform the user that processing is complete and the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()


@cl.on_message
async def main(message: cl.Message):
    """
    Processes incoming messages and generates a response using the LLM chain.
    This function also handles the retrieval of relevant document chunks.

    Args:
        message (cl.Message): The incoming message from the user.
    """
    # Retrieve the chain and prompt template from the user session
    chain = cl.user_session.get("chain")
    prompt_template = cl.user_session.get("prompt_template")
    global file_contents

    # Initialize callback handler for asynchronous operations
    cb = cl.AsyncLangchainCallbackHandler()

    # Load the existing conversation history
    context = chain.memory.load_memory_variables({})["chat_history"]

    # Generate the final prompt including the conversation history and the file contents
    prompt_content = prompt_template.format(context=context, file_contents=file_contents, question=message.content)

    # Log the generated prompt for debugging purposes
    print("Generated Prompt:")
    print(prompt_content)

    # Invoke the chain with the formatted prompt
    res = await chain.ainvoke({"question": prompt_content}, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]

    text_elements = []  # Initialize list to store text elements

    # Process source documents if available
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        # Add source references to the answer
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    # Return the final answer with sources if available
    await cl.Message(content=answer, elements=text_elements).send()
