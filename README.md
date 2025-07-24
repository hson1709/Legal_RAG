# Legal Consultation Chatbot

A chatbot built with RAG (Retrieval-Augmented Generation), designed to provide accurate legal advice based on Vietnamese law documents. The application uses advanced embedding models and vector databases to retrieve relevant legal information and generate contextual responses.

##  Features

- **Multi-Intent Handling**
  - **Lookup**: Retrieve legal information for a given topic or keyword
  - **Comparison**: Compare legal regulations across documents, time periods, or topics
  - **Analysis**: Summarize, break down, and reason about legal situations
- **Hybrid Search** (Vector + Keyword)
  - Combines semantic search (via vector embeddings) with lexical search (BM25-based keyword retriever)
  - Allows adjustable weights between vector and keyword scores for optimal performance across use cases
- **Reranker for accuracy improvement**
  - A CrossEncoder reranker re-scores retrieved documents for relevance to the query
  - Improves final document ranking before generating answers
- **RAG-based Question Answering**: Combines vector retrieval with generative answers
- **Streamlit Web Interface**: Easy-to-use chat-based frontend
- **Source Citations**: Display exact documents and article numbers used in answers
- **Conversation History**: Maintains chat history during session
- **Vietnamese Law Focus**: Expertly tuned for the Vietnamese legal system
- **Model Selection**: Choose between multiple LLM providers (e.g., OpenAI, Gemini)

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

##  Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd legal-consultation-chatbot
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add API Key Configuration
Create a .env file with your LLM credentials:

```bash
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key
```

## Running the Application

### 1. Start the Streamlit Application

```bash
streamlit run app.py
```

### 2. Access the Application

Open your web browser and navigate to:
```
http://localhost:8501
```

### 3. Using the Chatbot

1. **Enter your legal question** in the input field
2. **Select model** (GPT or Gemini) from the sidebar
3. **Select chunk size** (512 or 1024 tokens) from the sidebar
4. **Enter chunk dosc** for retriver from the sidebar
5. **Adjust hybrid search weight** between vector and keyword retrievers
6. **Click "Send"** to get your answer
7. **View source references** by expanding the reference section
8. **Clear chat history** using the sidebar button if needed

## Configuration

### Chunk Size Selection

- **512 tokens**: More precise retrieval, suitable for focused and specific legal queries
- **1024 tokens**: More comprehensive context, better for complex legal questions