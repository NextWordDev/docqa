import streamlit as st
import os
from PIL import Image
# Set the max file size limit (in MB)
st.set_page_config(page_title="File Uploader", page_icon=None, layout="centered", initial_sidebar_state="auto")

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


from llama_index import download_loader
from llama_index.node_parser import SimpleNodeParser
from llama_index import GPTSimpleVectorIndex
from llama_index import LLMPredictor, GPTSimpleVectorIndex, PromptHelper, ServiceContext
from langchain import OpenAI

SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
st.title("Private Document Q&A")

index_file = 'index.json'
# Initialize session state variables if not already set
if "history" not in st.session_state:
    st.session_state.history = []
if "prompt" not in st.session_state:
    st.session_state.prompt = ""
if "index" not in st.session_state:
    st.session_state.index = None
response_container = st.empty()
sidebar_placeholder = st.sidebar.container()

# Function to handle successful file upload
def handle_file_upload(uploaded_files):
    st.write(f"{len(uploaded_files)} file(s) uploaded successfully!")

def construct_index(uploaded_files):
    loader = SimpleDirectoryReader("./data", recursive=True, exclude_hidden=True)
    documents = loader.load_data()
    sidebar_placeholder.header('Current Processing Document:')

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

    max_input_size = 4096
    num_output = 256
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index = GPTSimpleVectorIndex.from_documents(
        documents, service_context=service_context
    )

    index.save_to_disk(index_file)    

    st.session_state.index = index

    for uploaded_file in uploaded_files:
        sidebar_placeholder.subheader(uploaded_file.name)
        sidebar_placeholder.write(documents[0].get_text()[:1000]+'...')

# Create the /data folder if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")

# Create the Streamlit app
st.title("File Uploader")

# File uploader component
uploaded_files = st.file_uploader("Choose one or more files", accept_multiple_files=True)

# If files have been uploaded
if uploaded_files:
    # Save the uploaded files to the /data folder
    for file in uploaded_files:
        file_name = os.path.join("data", file.name)
        with open(file_name, "wb") as f:
            f.write(file.getvalue())

    # Trigger the handler function
    construct_index(uploaded_files)
else:
    if os.path.exists(index_file):
        index = GPTSimpleVectorIndex.load_from_disk(index_file)
        st.session_state.index = index

        SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
        loader = SimpleDirectoryReader("./data", recursive=True, exclude_hidden=True)
        doc_filenames = os.listdir("./data")
        sidebar_placeholder.header('Documents Indexed:')
        for doc in doc_filenames:
            sidebar_placeholder.subheader(doc)

def handle_response():
    print(st.session_state.prompt)
    if st.session_state.prompt == '':
        return
    if st.session_state.index is None:
        return
    response = index.query(st.session_state.prompt)
    st.session_state.history.append(response)


st.text_input("Ask something: ", key='prompt', on_change=handle_response)


for response in st.session_state.history:
    st.write(response)