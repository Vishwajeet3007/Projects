import streamlit as st
from langgraph_database_backend import chatbot,retrive_all_threads
from langchain_core.messages import HumanMessage

# *********************************** Utility Functions ******************************

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    return chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values['messages']


# *********************************** Session Setup **********************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = None  # No thread yet

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrive_all_threads()  # Load existing threads from the database

# *********************************** Sidebar UI *************************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    st.session_state['thread_id'] = None
    st.session_state['message_history'] = []

st.sidebar.header('My Conversations')

# Show previous chats using first message as label
for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(thread_id):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages


# *********************************** Main Chat UI ***********************************

# Display existing messages
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # If it's the first message, use it as thread_id/title
    if st.session_state['thread_id'] is None:
        thread_id = user_input.strip()
        st.session_state['thread_id'] = thread_id
        add_thread(thread_id)

    # Add user message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # Stream assistant response
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            )
        )

    # Add assistant reply
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
