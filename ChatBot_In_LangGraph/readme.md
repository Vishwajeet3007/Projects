💬 LangGraph + Gemini AI Chatbot with Streamlit
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/Vishwajeet3007/Agentic-AI-Using-LangGraph/tree/main/ChatBot_In_LangGraph)

This repository contains a conversational AI chatbot built with LangGraph, Google's Gemini Pro model, and Streamlit. It demonstrates how to create a stateful, memory-enabled chatbot using a state graph. The project includes three different Streamlit frontends, showcasing basic, streaming, and multi-conversation (threading) capabilities — plus a new database-backed version for persistent storage.

✨ Features
Stateful Conversations: Utilizes LangGraph's StateGraph and InMemorySaver to manage conversation flow and maintain memory across interactions.

LLM Integration: Powered by Google's gemini-1.5-flash model through the langchain-google-genai package.

Multiple Frontend Examples:

streamlit_frontend.py: A basic, non-streaming interface.

streamlit_frontend_streaming.py: An improved UI with real-time, streaming responses.

streamlit_frontend_threading.py: An advanced UI supporting multiple, separate conversations (threads) that can be saved and revisited.

streamlit_frontend_database.py: New frontend using a database-backed backend.

Database-Backed Backend (langgraph_database_backend.py):
Implements persistent storage of conversation threads. Includes the function:

python
Copy
Edit
def retrieve_all_threads(): 
    """Retrieve all threads from the database."""
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)
Session State Initialization (streamlit_frontend_database.py):
Loads saved conversation threads into Streamlit session state on app startup:

python
Copy
Edit
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()  # Load existing threads from the database
Modular Code: Clean separation between backend graph logic and user-facing Streamlit applications.

🚀 How It Works
The application supports two main backend/frontend modes:

In-Memory Mode (Original):
Uses langgraph_backend.py with the three Streamlit frontends:

streamlit_frontend.py (basic, non-streaming)

streamlit_frontend_streaming.py (streaming)

streamlit_frontend_threading.py (streaming + multi-threading)

In this mode, conversation state is stored in memory and session state only.

Database-Backed Mode (New):
Uses langgraph_database_backend.py and streamlit_frontend_database.py. This mode persists conversation threads in a database via the checkpointer, enabling session persistence across app restarts and user switching.

The core of the application remains a state machine defined as a StateGraph which sends conversation messages to the Gemini LLM and stores responses.

📸 Chatbot Demo
Here’s a quick preview of the working chatbot UI:

<p align="center"> <img src="../ChatBot_In_LangGraph/ChatBot_OUTPUT.jpg" alt="Chatbot Demo" width="600"/> </p>
🗂️ Project Structure
bash
Copy
Edit
.
├── .env                            # Stores the GOOGLE_API_KEY
├── langgraph_backend.py            # Core LangGraph logic and Gemini model setup (in-memory)
├── langgraph_database_backend.py   # Extended backend with database support (new)
├── requirements.txt                # Project dependencies
├── streamlit_frontend.py           # Basic Streamlit UI (non-streaming, in-memory)
├── streamlit_frontend_streaming.py # Streamlit UI with streaming responses (in-memory)
├── streamlit_frontend_threading.py # Advanced UI with streaming & multi-conversation support (in-memory)
├── streamlit_frontend_database.py  # New Streamlit frontend with database backend support
🛠️ Setup and Usage
Follow these steps to run the chatbot on your local machine.

1. Prerequisites
Python 3.8+

A Google AI API Key

2. Clone the Repository
bash
Copy
Edit
git clone https://github.com/Vishwajeet3007/Agentic-AI-Using-LangGraph.git
cd Agentic-AI-Using-LangGraph/ChatBot_In_LangGraph
3. Install Dependencies
Create and activate a virtual environment, then install the required packages.

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
4. Configure Environment Variables
Create a .env file in the ChatBot_In_LangGraph directory and add your Google API key:

.env
Copy
Edit
GOOGLE_API_KEY="YOUR_GOOGLE_AI_API_KEY"
You can obtain an API key from Google AI Studio.

5. Run the Application
Choose one of the following to start the chatbot:

a) Run the In-Memory Version (original)
bash
Copy
Edit
# Advanced streaming + multi-threading
streamlit run streamlit_frontend_threading.py

# Or simpler streaming version
# streamlit run streamlit_frontend_streaming.py

# Or basic non-streaming version
# streamlit run streamlit_frontend.py
b) Run the Database-Backed Version (new)
bash
Copy
Edit
streamlit run streamlit_frontend_database.py
This runs the chatbot with database persistence of conversation threads.

Open your browser to the local URL provided by Streamlit (usually http://localhost:8501) to start chatting.

⚙️ Technology Stack
Backend: LangGraph, LangChain

LLM: Google Gemini 1.5 Flash

Frontend: Streamlit

Dependencies: python-dotenv, google-generativeai
