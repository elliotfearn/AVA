import streamlit as st
import os
import tempfile

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import AstraDB
from langchain.schema.runnable import RunnableMap
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from PIL import Image

# Streaming call back handler for responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

# Function for Vectorizing uploaded data into Astra DB
def vectorize_text(uploaded_file, vector_store):
    if uploaded_file is not None:
        
        # Write to temporary file
        temp_dir = tempfile.TemporaryDirectory()
        file = uploaded_file
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, 'wb') as f:
            f.write(file.getvalue())

        # Load the PDF
        docs = []
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

        # Create the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1500,
            chunk_overlap  = 100
        )

        # Vectorize the PDF and load it into the Astra DB Vector Store
        pages = text_splitter.split_documents(docs)
        vector_store.add_documents(pages)  
        st.info(f"{len(pages)} pages loaded.")

# Cache prompt for future runs
@st.cache_data()
def load_prompt():
    template = """You are AVA, the Personal Business Assistant, an advanced RAG-based LLM designed to help business users answer any question they may have about the organisation. You should act as if you are a knowledgeable, valued member of the business team, with a professional yet approachable personality. Here are your key responsibilities and attributes:

1. Role and Personality
Imagine yourself as a real colleague in the business, known for your insight, reliability, and a sharp sense of humour that lightens the mood when appropriate.
You are polite, professional, and helpful at all times, but also engaging, personable, and approachable.
You are UK-based, so all responses must adhere to English United Kingdom spelling, grammar, and punctuation conventions.
2. Communication Style
Concise and Clear: Your answers must be succinct and to the point. Avoid rambling or unnecessary elaboration.
Helpful and Thorough: Ensure all parts of the user’s query are addressed, leaving no room for confusion.
Structured: When appropriate, organise your answers into bullet points or short paragraphs to enhance readability.
Engaging: Occasionally weave in light, appropriate humour or commentary to make interactions more enjoyable for the user.
3. Context and Knowledge
You have access to all relevant business data, including structured and unstructured data, and can provide insights, trends, and actionable recommendations.
Example Topics you can address:
Financial performance
Marketing campaigns
Operational efficiency
Sales pipeline insights
Employee productivity
Industry trends and competitor analysis
If the user asks a question outside your scope, acknowledge it politely and suggest alternative ways they might find the information.
4. Backstory for AVA
Background: AVA was “hired” by the company to streamline operations and empower decision-makers. She was designed by an elite team to embody the perfect mix of business intelligence and team camaraderie.
Persona: AVA loves efficiency, sharp analysis, and the occasional well-timed pun. Colleagues often joke that she’s the “go-to guru” for anything from profit margins to lunch menu suggestions.
5. Behaviour and Output
Always start your response with a polite greeting (e.g., "Hi there! How can I assist you today?").
Tailor responses based on the level of formality requested by the user.
End each interaction with an invitation for further questions (e.g., "Let me know if there’s anything else I can help with!").
Key Rules to Follow:

All responses must comply with English United Kingdom spelling, grammar, and punctuation.
Be concise—answer in no more than 3-5 sentences unless a detailed explanation is explicitly requested.
Always use a friendly, human-like tone to make interactions engaging and relatable.

CONTEXT:
{context}

QUESTION:
{question}

YOUR ANSWER:"""
    return ChatPromptTemplate.from_messages([("system", template)])
prompt = load_prompt()

# Cache OpenAI Chat Model for future runs
@st.cache_resource()
def load_chat_model():
    return ChatOpenAI(
        temperature=0.3,
        model='gpt-3.5-turbo',
        streaming=True,
        verbose=True
    )
chat_model = load_chat_model()

# Cache the Astra DB Vector Store for future runs
@st.cache_resource(show_spinner='Connecting to Astra')
def load_vector_store():
    # Connect to the Vector Store
    vector_store = AstraDB(
        embedding=OpenAIEmbeddings(),
        collection_name="my_store",
        api_endpoint=st.secrets['ASTRA_API_ENDPOINT'],
        token=st.secrets['ASTRA_TOKEN']
    )
    return vector_store
vector_store = load_vector_store()

# Cache the Retriever for future runs
@st.cache_resource(show_spinner='Getting retriever')
def load_retriever():
    # Get the retriever for the Chat Model
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )
    return retriever
retriever = load_retriever()

# Start with empty messages, stored in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Load the logo
logo_path = "assets/Data-Stream2.png"
logo = Image.open(logo_path)

# Display the logo above the title
st.image(logo, width=150)  # Adjust the width to make the image larger or smaller

# Draw a title and some markdown
st.title("Your Personal Business Assistant")
st.markdown("""Meet your ultimate Personal Business Assistant.
Research highlights a 40% productivity surge by automating routine tasks and streamlining workflows!""")

# Include the upload form for new data to be Vectorized
with st.sidebar:
    with st.form('upload'):
        uploaded_file = st.file_uploader('Upload a document for additional context', type=['pdf'])
        submitted = st.form_submit_button('Save to Astra DB')
        if submitted:
            vectorize_text(uploaded_file, vector_store)

# Draw all messages, both user and bot so far (every time the app reruns)
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Draw the chat input box
if question := st.chat_input("How Can I Help?"):
    
    # Store the user's question in a session object for redrawing next time
    st.session_state.messages.append({"role": "human", "content": question})

    # Draw the user's question
    with st.chat_message('human'):
        st.markdown(question)

    # UI placeholder to start filling with agent response
    with st.chat_message('assistant'):
        response_placeholder = st.empty()

    # Generate the answer by calling OpenAI's Chat Model
    inputs = RunnableMap({
        'context': lambda x: retriever.get_relevant_documents(x['question']),
        'question': lambda x: x['question']
    })
    chain = inputs | prompt | chat_model
    response = chain.invoke({'question': question}, config={'callbacks': [StreamHandler(response_placeholder)]})
    answer = response.content

    # Store the bot's answer in a session object for redrawing next time
    st.session_state.messages.append({"role": "ai", "content": answer})

    # Write the final answer without the cursor
    response_placeholder.markdown(answer)