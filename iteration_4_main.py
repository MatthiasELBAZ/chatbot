import streamlit as st
import datetime
import pandas as pd
from pathlib import Path
import tempfile
import os
from iteration_4_python import *



with st.sidebar:
    session_id = st.text_input("Session ID", key="session_id")
    llm_name_model = st.selectbox('Pick a llm model',('openai', 'anthropic', 'mistralai'))
    embedding_name_model = st.selectbox('Pick a embedding model',('openai','none'))        
    llm_api_key = st.text_input("LLM API Key", key="chatbot_api_key", type="password")

    if session_id and llm_name_model and llm_api_key and embedding_name_model: 

        llm = select_llm_model(llm_name_model)
        embedding = select_embedding_model(embedding_name_model)
        store_history = {
        session_id:
        {
            "ongo":ChatMessageHistory(), 
            "full":ChatMessageHistory()
        }
    }

    else:
        st.error("Please enter the LLM API key.")


    uploaded_files = st.file_uploader("Choose a User folder and upload files", accept_multiple_files=True, type=['txt', 'csv', 'png', 'jpg', 'pdf'])

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
            if 'journal.csv' in uploaded_file.name:
                journal_path_input = os.path.join(temp_dir, uploaded_file.name)
            elif 'formulaire.csv' in uploaded_file.name:
                formulaire_path_input = os.path.join(temp_dir, uploaded_file.name)
            elif 'history.csv' in uploaded_file.name:
                history_path_input = os.path.join(temp_dir, uploaded_file.name)


# Main chat interface
st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")

if st.button('Start Chat'):

    # load documents
    formulaire_docs = load_csv_from_directory(formulaire_path_input.split(".")[0])
    journal_docs = load_csv_from_directory(journal_path_input.split(".")[0])
    history_docs = load_csv_from_directory(history_path_input.split(".")[0])
    st.success("CSV files loaded successfully.")

    # generate user profile
    user_profile = user_profile_generation(llm, formulaire_docs, journal_docs)
    st.success("User profile generated successfully.")
    st.write(f"User Profile: {user_profile}")

    # create time weighted vector store retriever
    faiss_time_retriever = faiss_time_retriever(embedding, journal_docs)
    st.success("Time weighted vector store retriever created successfully.")

    # create rag chain chatbot
    document_chain = get_document_chain(llm)
    document_chain_with_message_history = get_document_chain_with_message_history(document_chain, store_history)
    retrieval_document_chain_with_message_history = et_retrieval_document_chain_with_message_history(document_chain_with_message_history, time_retriever, llm, store_history)
    st.success("Rag chain chatbot created successfully.")




