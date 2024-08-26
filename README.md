# Wikipedia Article Q&A Chatbot

## Project Description
This project is a Q&A chatbot designed to interact with Wikipedia articles. By inputting a Wikipedia URL, users can engage in a chat session where they can ask detailed questions about the article's content. The chatbot processes the article, segments it into manageable chunks, and uses these chunks to provide accurate, context-aware responses. This application is built using Python and leverages technologies like Streamlit, LangChain, and AzureChatOpenAI to handle the interactive sessions and content retrieval.

## Features
- **URL Input**: Users can input any Wikipedia URL for querying.
- **Interactive Q&A**: Direct interaction with the chatbot to ask questions about the article.
- **Context Management**: Stores conversation history to maintain context and improve response accuracy.
- **Caching**: Saves processed vector data to avoid reprocessing for repeated queries.

## Technologies Used
- **Python**: Primary programming language for backend development.
- **Streamlit**: For creating the web interface.
- **FAISS**: For efficient similarity search and clustering of vectors.
- **AzureChatOpenAI**: For generating responses based on the text.
- **HuggingFace BgeEmbeddings**: For embedding text into vectors.
- **os, hashlib, pickle**: For file and data management.

## Installation
Ensure you have Python installed on your machine. Then clone this repository and install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/your-username/your-repository-name.git

# Navigate to the project directory
cd your-repository-name

# Install required Python packages
pip install -r requirements.txt
```

```bash
# Setup .env

USER_ID = 'Your User ID'
DEPLOYMENT_NAME = 'Your Deployment Name'
MODEL_NAME = 'Your Model Name'

OPENAI_API_KEY = 'Your API Key'
AZURE_OPENAI_ENDPOINT = 'Openai Endpoint'
OPENAI_API_TYPE = 'API Type'
OPENAI_API_VERSION = 'Your version'
```

## Usage
To run the chatbot, execute the Streamlit application using the following command:

```bash
streamlit run app.py
```

## Architecture Overview
### 1. Scraping Text from Input URL
Using Selenium to scrape the content of any Wikipedia page provided by a URL.

### 2. Text Processing
Breaking down the article into smaller, manageable chunks with a recursive character text splitter.

### 3. Vector Store Management
Utilizing FAISS to store and retrieve processed text in vector format for efficient retrieval.

### 4. Conversation Handling
Managing user interactions, maintaining chat history, and generating responses using the AzureChatOpenAI model.

### 5. User Interface
Built with Streamlit to provide an easy-to-use interface for users to input URLs and questions, and to display responses.

## Flow Diagram
![Flow Diagram](https://github.com/kumawatsr/Wikipedia-Chatbot/blob/main/templates/flowchart.png)

## Contributing
Contributions are welcome! If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## License
This project is licensed under the MIT License.
