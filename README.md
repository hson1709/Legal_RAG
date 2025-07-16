# Legal Consultation Chatbot

A chatbot built with RAG (Retrieval-Augmented Generation), designed to provide accurate legal advice based on Vietnamese law documents. The application uses advanced embedding models and vector databases to retrieve relevant legal information and generate contextual responses.

## Features

- **RAG-based Question Answering**: Utilizes retrieval-augmented generation for accurate legal responses
- **Interactive Web Interface**: Built with Streamlit for easy user interaction
- **Conversation History**: Maintains chat history throughout the session
- **Source Citations**: Provides references to legal documents used in responses
- **Vietnamese Law Focus**: Specialized in Vietnamese legal system and regulations

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
2. **Select chunk size** (512 or 1024 tokens) from the sidebar
3. **Click "Send"** to get your answer
4. **View source references** by expanding the reference section
5. **Clear chat history** using the sidebar button if needed

## Configuration

### Chunk Size Selection

- **512 tokens**: More precise retrieval, suitable for focused and specific legal queries
- **1024 tokens**: More comprehensive context, better for complex legal questions

### Model Configuration

The application uses the following components:
- **Embedding Model**: For document and query vectorization
- **Vector Database**: ChromaDB for efficient similarity search
- **Language Model**: For generating contextual responses
- **Prompt Template**: Optimized for legal consultation scenarios
