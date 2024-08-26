import os
import hashlib
import pickle
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from htmlTemplates import css, bot_template, user_template  # Import HTML templates for Streamlit UI
from scraper import scrape_wikipedia  # Import Wikipedia scraper function
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class TextProcessor:
    """Class to handle text processing and chunking."""

    def __init__(self, separators=["\n\n\n","\n\n", "\n"], chunk_size=1500, chunk_overlap=300):
        # Initialize the text splitter with provided configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def get_text_chunks(self, text):
        """Split the input text into smaller chunks."""
        return self.text_splitter.split_text(text)

class VectorStoreManager:
    """Class to manage the FAISS vector store."""

    def __init__(self):
        # Directory to store vector files
        self.directory = 'vectorstores'
        os.makedirs(self.directory, exist_ok=True)

    def get_vectorstore(self, text_chunks, url):
        """Retrieve or create a vector store for the given text chunks and URL."""
        # Generate a unique filename based on the URL hash
        filename = hashlib.sha256(url.encode('utf-8')).hexdigest()[:10] + '.pkl'
        filepath = os.path.join(self.directory, filename)

        if os.path.exists(filepath):
            # Load vector store from file if it exists
            with open(filepath, 'rb') as f:
                vectorstore = pickle.load(f)
            st.write("Using cached data.")
        else:
            # Create a new vector store if it doesn't exist
            model_name = "BAAI/bge-base-en"
            model_kwargs = {"device": "cpu"}
            encode_kwargs = {"normalize_embeddings": True}
            embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
            )
        
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            # Save the vector store to file
            with open(filepath, 'wb') as f:
                pickle.dump(vectorstore, f)
                st.write(f"Data saved in: {filename}")
                
        return vectorstore

class ConversationHandler:
    """Class to handle the conversational chain with the language model."""

    def __init__(self, user_id, deployment_name):
        # Initialize the language model with Azure OpenAI
        self.llm = AzureChatOpenAI(
            default_headers={"User-Id": user_id},
            temperature=0.0,
            deployment_name=deployment_name,
        )
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        
        # Define the prompt template for the conversation
        self.template = """
        Instructions:
        1. Carefully read the provided context.
        2. Answer the question based on the information within the context.
        3. If the answer is not available in the context, explicitly state "Sorry, I couldn't find the answer in the given context."
        4. Do not provide any information that is not present in the context.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

    def get_conversation_chain(self, vectorstore):
        """Create a conversational retrieval chain using the vector store."""
        prompt_template = PromptTemplate(template=self.template, input_variables=["context", "question"])
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )

class StreamlitApp:
    """Main class for the Streamlit application."""

    def __init__(self):
        self.text_processor = TextProcessor()
        self.vectorstore_manager = VectorStoreManager()
        self.conversation_handler = ConversationHandler(os.getenv("USER_ID"), os.getenv("DEPLOYMENT_NAME"))

    def handle_userinput(self, user_question):
        """Handle user input and update the conversation."""
        response = st.session_state.conversation({"query": user_question})
        response = response['result']
        
        st.session_state.chat_history.append(user_question)
        st.session_state.chat_history.append(response)
        
        # Display chat history in the UI
        for i, message in enumerate(st.session_state.chat_history):
            template = user_template if i % 2 == 0 else bot_template
            st.write(template.replace("{{MSG}}", str(message)), unsafe_allow_html=True)
            

    def run(self):
        """Run the main application."""
        st.set_page_config(page_title="Chat with Wikipedia Articles", page_icon=":books:")
        st.write(css, unsafe_allow_html=True)

        # Initialize session state variables if not already set
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.header("Chat with Wikipedia Articles :books:")
        user_question = st.text_input("Ask a question about your article:")
        if user_question:
            self.handle_userinput(user_question)

        with st.sidebar:
            url = st.text_input("Enter the Wikipedia URL:")
            if st.button("Process"):
                with st.spinner("Processing"):
                    # Scrape the Wikipedia article if the vector store file does not exist
                    scraped_text = scrape_wikipedia(url)
                    if scraped_text == "Invalid URL":
                        st.error("Invalid URL. Please enter a valid Wikipedia URL.")
                        return
                    else:
                        raw_text = scraped_text if not os.path.exists(os.path.join('vectorstores', hashlib.sha256(url.encode('utf-8')).hexdigest()[:10] + '.pkl')) else None
                    text_chunks = self.text_processor.get_text_chunks(raw_text) if raw_text else []
                    vectorstore = self.vectorstore_manager.get_vectorstore(text_chunks, url)
                    # Initialize the conversation chain
                    st.session_state.conversation = self.conversation_handler.get_conversation_chain(vectorstore)

if __name__ == '__main__':
    app = StreamlitApp()
    app.run()
