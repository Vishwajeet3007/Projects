# AI Study Buddy
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/Vishwajeet3007/Projects/tree/main/ai-study-buddy)

This project provides a "Study Buddy" application built with Streamlit and LangChain. It allows you to upload a PDF document (like a textbook, syllabus, or research paper) and ask questions about its content. The application uses a Retrieval-Augmented Generation (RAG) pipeline to provide context-aware answers.

The repository contains two versions of the Streamlit application:
*   `app.py`: A streamlined version using Google Gemini for both embeddings and the chat model.
*   `app1.py`: A more advanced version with features like quiz generation, choice between OpenAI and Gemini models, local HuggingFace embeddings, and optional voice I/O and translation.

## Key Features

*   **PDF Q&A:** Upload any PDF and chat with it to get answers, summaries, or explanations.
*   **RAG Pipeline:** Uses FAISS for efficient similarity search to find relevant document chunks to answer your questions.
*   **Flexible LLM & Embedding Support:**
    *   **`app.py`:** Uses Google Gemini for both chat and embeddings.
    *   **`app1.py`:** Uses local `sentence-transformers` for embeddings (free, no API key needed) and allows switching between Google Gemini and OpenAI (GPT-4o-mini) for chat.
*   **Quiz Generation:** `app1.py` can generate multiple-choice quizzes on a specific topic from the document to test your knowledge.
*   **Voice & Translation (Optional in `app1.py`):**
    *   **Voice Input:** Use your microphone to ask questions (via `speech_recognition`).
    *   **Voice Output:** Hear the AI's answers spoken aloud (via `gTTS`).
    *   **Translation:** Translate answers between English and Hindi (via `transformers`).

## Architecture

The application follows a standard RAG pattern:

1.  **PDF Loading:** The text content is extracted from the uploaded PDF file using `pypdf`.
2.  **Text Splitting:** The extracted text is divided into smaller, manageable chunks using `LangChain`'s `RecursiveCharacterTextSplitter`.
3.  **Embedding:** Each text chunk is converted into a numerical vector (an embedding). This is done using either Google Gemini Embeddings or a local HuggingFace Sentence Transformer model.
4.  **Vector Storage:** The embeddings are stored in a `FAISS` vector store, which allows for fast and efficient retrieval of chunks that are semantically similar to a user's query.
5.  **Retrieval:** When a user asks a question, the application embeds the query and uses FAISS to find the most relevant text chunks from the original document.
6.  **Generation:** The user's query and the retrieved text chunks are passed as context to a powerful LLM (Gemini or GPT), which generates a comprehensive answer.
7.  **Frontend:** `Streamlit` provides the interactive web user interface.

## Setup and Installation

Follow these steps to set up and run the project locally.

**1. Clone the Repository**
```bash
git clone https://github.com/vishwajeet3007/Projects.git
cd Projects/ai-study-buddy
```

**2. Install Dependencies**
It is recommended to use a virtual environment.
```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt
```

**3. Set Up API Keys**
*   Create a file named `.env` in the `ai-study-buddy` directory by copying the example:
    ```bash
    cp .env.example .env
    ```
*   Edit the `.env` file and add your API keys. You will need a Gemini API key for `app.py` and `app1.py`. An OpenAI key is only needed if you wish to use the OpenAI model in `app1.py`.

    ```env
    # Get your key from Google AI Studio: https://aistudio.google.com/app/apikey
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"

    # Get your key from OpenAI Platform: https://platform.openai.com/api-keys
    # OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    ```

## How to Run

You can either build the document index on-the-fly through the web app or pre-build it using a script for larger documents.

### Method 1: Interactive Mode (Easiest)

Run one of the Streamlit apps directly. The app will prompt you to upload a PDF, which it will then index automatically.

**For the simplified Gemini-based app:**
```bash
streamlit run app.py
```

**For the advanced app with more features:**
```bash
streamlit run app1.py
```

Once the app is running, use the sidebar to upload your PDF. After indexing is complete, you can start asking questions.

### Method 2: Pre-build the Index

This method is useful for large PDFs to avoid re-indexing every time you start the app.

1.  Place your PDF file in the `ai-study-buddy` directory (e.g., `my_document.pdf`).
2.  Edit the `build_index.py` script to point to your PDF file name.
    ```python
    # in build_index.py
    pdf_path = "my_document.pdf"  # <-- Change this to your PDF file
    ```
3.  Run the script to create the `faiss_index` directory.
    ```bash
    python build_index.py
    ```
4.  Now, run either Streamlit app. It will automatically detect and load the pre-built index.
    ```bash
    streamlit run app.py