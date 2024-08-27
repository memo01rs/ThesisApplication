import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl


# This function is triggered when the chat session starts
@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a PDF file
    while not files:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=100,
            timeout=180,
        ).send()

    file = files[0]  # Use the first uploaded file

    # Notify the user that processing has started
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Read and extract text from the PDF file
    pdf = PyPDF2.PdfReader(file.path)
    pdf_text = "".join([page.extract_text() for page in pdf.pages])

    # Split the text into chunks for easier processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=180)
    text_chunks = text_splitter.split_text(pdf_text)
    metadatas = [{"source": f"Page {i + 1}"} for i in range(len(text_chunks))]

    # Create embeddings for the text chunks and build the vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = await cl.make_async(Chroma.from_texts)(
        text_chunks, embeddings, metadatas=metadatas
    )

    # Initialize memory to maintain conversation history
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Du bist eine KI, die sich auf die Analyse und Verarbeitung von Reglementen spezialisiert hat. "
                "Du musst den Anweisungen des Nutzers folgen und detaillierte, präzise und kontextuell relevante Antworten geben. "
                "Du darfst keine Informationen 'halluzinieren' oder erfinden, die nicht im Dokument enthalten sind. "
                "Wenn das Dokument und die Frage auf Deutsch verfasst sind, gib deine Antwort auf Deutsch.",
            ),
            (
                "user",
                "Kontext:\n{context}\n\nFrage: {question}",
            ),
        ]
    )

    # Create the processing chain combining the prompt and the language model
    chain = RunnablePassthrough() | prompt | ChatOllama(model="mistral-nemo:latest", temperature=0.2)

    # Store the chain and the retriever in the user session
    cl.user_session.set("chain", chain)
    cl.user_session.set("retriever", docsearch.as_retriever())

    # Notify the user that processing is complete
    msg.content = f"Processing of `{file.name}` completed. You can now ask questions!"
    await msg.update()


# This function handles incoming messages from the user
@cl.on_message
async def handle_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    retriever = cl.user_session.get("retriever")

    # Retrieve relevant contexts from the vector store based on the user's question
    relevant_contexts = retriever.get_relevant_documents(message.content)
    context = "\n\n".join([doc.page_content for doc in relevant_contexts])

    # Generate the response using the processing chain
    response = await chain.ainvoke({"context": context, "question": message.content})
    answer = response.content

    # Send the response back to the user
    await cl.Message(content=answer).send()
