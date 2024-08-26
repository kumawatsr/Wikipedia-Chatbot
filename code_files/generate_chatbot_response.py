import os
import hashlib
import pickle
import pandas as pd
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from scraper import scrape_wikipedia

# Load environment variables
load_dotenv()

class TextProcessor:
    def __init__(self, separators=["\n", "\n\n", "\n\n\n"], chunk_size=1500, chunk_overlap=300):
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def get_text_chunks(self, text):
        return self.text_splitter.split_text(text)

class VectorStoreManager:
    def __init__(self):
        self.directory = 'vectorstores'
        os.makedirs(self.directory, exist_ok=True)

    def get_vectorstore(self, text_chunks, url):
        filename = hashlib.sha256(url.encode('utf-8')).hexdigest()[:10] + '.pkl'
        filepath = os.path.join(self.directory, filename)

        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                vectorstore = pickle.load(f)
            print("Using cached data.")
        else:
            model_name = "BAAI/bge-base-en"
            model_kwargs = {"device": "cpu"}
            encode_kwargs = {"normalize_embeddings": True}
            embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
            )
        
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            with open(filepath, 'wb') as f:
                pickle.dump(vectorstore, f)
                print(f"Data saved in: {filename}")
                
        return vectorstore

class ConversationHandler:
    def __init__(self, user_id, deployment_name):
        self.llm = AzureChatOpenAI(
            default_headers={"User-Id": user_id},
            temperature=0.0,
            deployment_name=deployment_name,
        )
        self.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        
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
        prompt_template = PromptTemplate(template=self.template, input_variables=["context", "question"])
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )

class ExcelProcessor:
    def __init__(self, file_path, sheet_name):
        self.file_path = file_path
        self.sheet_name = sheet_name

    def process_excel(self):
        df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
        save_file_name = "test_cases_with_chatbot_answer.xlsx"

        unique_urls = df['URL'].unique()

        for url in unique_urls:
            article_text = scrape_wikipedia(url)
            
            text_processor = TextProcessor()
            text_chunks = text_processor.get_text_chunks(article_text)

            vectorstore_manager = VectorStoreManager()
            vectorstore = vectorstore_manager.get_vectorstore(text_chunks, url)

            conversation_handler = ConversationHandler(os.getenv("USER_ID"), os.getenv("DEPLOYMENT_NAME"))
            qa_chain = conversation_handler.get_conversation_chain(vectorstore)

            url_rows = df[df['URL'] == url]

            for index, row in url_rows.iterrows():
                question = row['Question']
                response = qa_chain({"query": question})
                df.at[index, 'Chatbot Answer'] = response['result']
        
        df.to_excel(save_file_name, index=False)
        print(f"Updated test cases saved to {save_file_name}")

if __name__ == "__main__":
    file_path = "Evaluation-Code-Assignment/testing_results/test_cases.xlsx"  
    sheet_name = "Sheet1"
    excel_processor = ExcelProcessor(file_path, sheet_name)
    excel_processor.process_excel()
