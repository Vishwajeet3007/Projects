# 💬 LangGraph + Gemini AI Chatbot with Streamlit
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/Vishwajeet3007/Agentic-AI-Using-LangGraph/tree/main/ChatBot_In_LangGraph)

This repository contains a conversational AI chatbot built with LangGraph, Google's Gemini Pro model, and Streamlit. It demonstrates how to create a stateful, memory-enabled chatbot using a state graph. The project includes three different Streamlit frontends, showcasing basic, streaming, and multi-conversation (threading) capabilities.

## ✨ Features

-   **Stateful Conversations**: Utilizes LangGraph's `StateGraph` and `InMemorySaver` to manage conversation flow and maintain memory across interactions.
-   **LLM Integration**: Powered by Google's `gemini-1.5-flash` model through the `langchain-google-genai` package.
-   **Multiple Frontend Examples**:
    -   `streamlit_frontend.py`: A basic, non-streaming interface.
    -   `streamlit_frontend_streaming.py`: An improved UI with real-time, streaming responses.
    -   `streamlit_frontend_threading.py`: An advanced UI supporting multiple, separate conversations (threads) that can be saved and revisited.
-   **Modular Code**: Clean separation between the backend graph logic (`langgraph_backend.py`) and the user-facing Streamlit applications.

## 🚀 How It Works

The application's core is a simple state machine defined in `langgraph_backend.py`.

1.  **State Graph**: A `StateGraph` is defined with a single state, `ChatState`, which holds the list of messages in the conversation.
2.  **Chat Node**: A single node, `chat_node`, takes the current state (messages), passes them to the Gemini LLM, and returns the AI's response.
3.  **Memory**: LangGraph's `InMemorySaver` acts as a checkpointer. It saves the state of the conversation, allowing it to be recalled using a unique `thread_id`. This is the mechanism that provides memory and enables multi-conversation support.
4.  **Frontend**: The Streamlit applications provide a user interface to interact with the graph. The most advanced version, `streamlit_frontend_threading.py`, allows users to create and switch between different conversation threads, each with its own persistent history for the session.


## 📸 Chatbot Demo

Here’s a quick preview of the working chatbot UI:

<p align="center">
  <img src="../ChatBot_In_LangGraph/ChatBot_OUTPUT.jpg" alt="Chatbot Demo" width="600"/>
</p>

## 🗂️ Project Structure

```
.
├── .env                  # Stores the GOOGLE_API_KEY
├── langgraph_backend.py    # Core LangGraph logic and Gemini model setup
├── requirements.txt      # Project dependencies
├── streamlit_frontend.py # Basic Streamlit UI (non-streaming)
├── streamlit_frontend_streaming.py # Streamlit UI with streaming responses
└── streamlit_frontend_threading.py # Advanced UI with streaming & multi-conversation support
```

## 🛠️ Setup and Usage

Follow these steps to run the chatbot on your local machine.

### 1. Prerequisites

-   Python 3.8+
-   A Google AI API Key

### 2. Clone the Repository

```bash
git clone https://github.com/Vishwajeet3007/Agentic-AI-Using-LangGraph.git
cd Agentic-AI-Using-LangGraph/ChatBot_In_LangGraph
```

### 3. Install Dependencies

Create and activate a virtual environment, then install the required packages.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a file named `.env` in the `ChatBot_In_LangGraph` directory and add your Google API key:

```.env
GOOGLE_API_KEY="YOUR_GOOGLE_AI_API_KEY"
```

You can obtain an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 5. Run the Application

You can run any of the three frontend applications. The most feature-complete version is `streamlit_frontend_threading.py`.

```bash
# Recommended: Run the advanced version with streaming and multi-conversation support
streamlit run streamlit_frontend_threading.py

# To run the simpler streaming version:
# streamlit run streamlit_frontend_streaming.py

# To run the basic non-streaming version:
# streamlit run streamlit_frontend.py
```

Open your browser to the local URL provided by Streamlit (usually `http://localhost:8501`) to start chatting.

## ⚙️ Technology Stack

-   **Backend**: LangGraph, LangChain
-   **LLM**: Google Gemini 1.5 Flash
-   **Frontend**: Streamlit
-   **Dependencies**: `python-dotenv`, `google-generativeai`
