import streamlit as st
import datetime
import pandas as pd
from pathlib import Path
import tempfile
import os
from iteration_1_python import *

openai_api_key = None

# Sidebar for API key input and relevant links
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    session_id = st.text_input("Session ID", key="session_id")
    uploaded_files = st.file_uploader("Choose a folder and upload files", accept_multiple_files=True, type=['txt', 'csv', 'png', 'jpg', 'pdf'])

    # Create a temporary directory to save uploaded files
    temp_dir = tempfile.mkdtemp()

    # Optionally list uploaded files and save them to the temporary directory
    if uploaded_files:
        st.write("Uploaded Files:")
        for uploaded_file in uploaded_files:
            # Write file to the temporary directory
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.write(f"{uploaded_file.name} saved at {file_path}")

# Initialize variables from uploaded files
journal_path_input, formulaire_path_input, history_path_input = None, None, None
for uploaded_file in uploaded_files:
    if 'journal.csv' in uploaded_file.name:
        journal_path_input = os.path.join(temp_dir, uploaded_file.name)
    elif 'formulaire.csv' in uploaded_file.name:
        formulaire_path_input = os.path.join(temp_dir, uploaded_file.name)
    elif 'history.csv' in uploaded_file.name:
        history_path_input = os.path.join(temp_dir, uploaded_file.name)

# Setup your LLM and other components
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, api_key=openai_api_key)
summarizer = load_summarize_chain(llm, chain_type="stuff")
new_user_summary = None
conversational_rag_chain = None

# Implement the button functionalities
date_today = get_today()
if st.button("Make Summary"):
    if formulaire_path_input :
        user_summary = create_user_summary(formulaire_path_input, summarizer)
        journal_summary = create_journal_summary(journal_path_input, summarizer)
        new_user_summary = modify_user_summary_with_journal(user_summary, journal_summary, llm)
        st.session_state['new_user_summary'] = new_user_summary
        st.write(f"User Summary: {new_user_summary}")

if st.button("Set Conversational Rag Chain"):
    if history_path_input:
        time_retriever = create_time_weighted_vector_store_retriever(history_path_input)
        rag_chain = create_rag_chain(llm, time_retriever)
        conversational_rag_chain = create_conversational_rag_chain(rag_chain)
        st.session_state['conversational_rag_chain'] = conversational_rag_chain
        st.success("Conversational Rag Chain is set")

# Main chat interface
st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")
if st.button('Start Chat'):
    st.session_state['chat_active'] = True
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

if st.button('End Chat'):
    st.session_state['chat_active'] = False
    if 'messages' in st.session_state:
        new_journal_message = summary_chat_history(llm, history_store)
        update_journal(new_journal_message, date_today, journal_path_input)
        update_chat_history(history_store, date_today)
        st.session_state['messages'] = []

# Chat interaction
if st.session_state.get('chat_active', False):
    for msg in st.session_state['messages']:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Your message"):
        new_user_summary = st.session_state['new_user_summary']
        conversational_rag_chain = st.session_state['conversational_rag_chain']

        if not all([openai_api_key, session_id, journal_path_input, formulaire_path_input, history_path_input, new_user_summary, conversational_rag_chain]):
            st.error("Please make sure all configurations are properly set.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            response = chat_bot_response(conversational_rag_chain, prompt, date_today, new_user_summary, session_id)
            st.session_state.messages.append({"role": "assistant", "content": response})
