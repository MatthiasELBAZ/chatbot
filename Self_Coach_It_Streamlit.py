
from Self_Coach_It_Classes import *
import streamlit as st

def handle_chat(prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response, cb = st.session_state['retrieval_document_chain_memory'].run_chat(prompt)
    answer = response['answer']
    st.session_state['messages'].append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)

def chat_interface():
    if "messages" in st.session_state:
        for msg in st.session_state['messages']:
            st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Your message"):
        handle_chat(prompt)

def end_chat(user_directory, user_id, session_id, store_history, llm):
    new_journal = NewJournal(llm, user_id, session_id, store_history)
    new_journal.get_new_journal()
    new_journal_text = new_journal.new_journal

    update_store = UpdateStore(user_directory + 'journal.csv', user_directory + 'history.csv', new_journal_text, store_history, user_id, session_id)
    update_store.update_store_journal_csv()
    update_store.update_store_chat_history()

    st.session_state['chat_active'] = False
    st.write("Chat ended and data updated.")

def main():
    st.sidebar.header("User Configuration")
    # User and Session ID
    user_id = st.sidebar.text_input("User ID", key="user_id")
    session_id = st.sidebar.text_input("Session ID", key="session_id")

    # Directory and API Key
    user_directory = st.sidebar.text_input("User Directory")
    api_key = st.sidebar.text_input("LLM API Key", key="chatbot_api_key", type="password")

    # Model Selection
    llm_name_model = st.sidebar.selectbox("LLM Model", ["openai", "anthropic", "mistralai"])
    embedding_name_model = st.sidebar.selectbox("Embedding Model", ["openai", "anthropic", "mistralai"])

    # Load Models and Documents
    if st.sidebar.button("Initialize"):
        with st.spinner('Loading models and documents...'):
            load_models = LoadModels(api_key, llm_name_model, embedding_name_model)
            llm = load_models.select_llm_model()
            embedding = load_models.select_embedding_model()

            user_session = UserSessionStoreHistory(user_id, session_id)
            user_session.initialize_store_history()
            store_history = user_session.store_history

            user_loader = UserDirectoryLoader(user_directory)
            formulaire_docs = user_loader.load_csv_from_directory('formulaire.csv')
            journal_docs = user_loader.load_csv_from_directory('journal.csv')

            # Generate User Profile and Initialize Time Retriever
            user_profile = UserProfile(llm, formulaire_docs, journal_docs)
            user_profile.user_profile_generation()
            user_profile_text = user_profile.user_profile
            st.write(user_profile_text)

            time_retriever = TimeWeightedRetriever(embedding, "faiss")
            time_retriever.get_vectorstore()
            time_retriever.get_time_retriever()
            time_retriever.time_retriever_add_documents(journal_docs)
            faiss_time_retriever = time_retriever.time_retriever

            # Initialize Chat Document Chain
            retrieval_document_chain_memory = RetrievalDocumentChainMemory(
                llm, 
                faiss_time_retriever, 
                user_profile_text, 
                user_id, 
                session_id, 
                store_history
            )
            retrieval_document_chain_memory.get_document_chain()
            retrieval_document_chain_memory.get_document_chain_with_message_history()
            retrieval_document_chain_memory.get_retrieval_document_chain_with_message_history()

            if 'retrieval_document_chain_memory' not in st.session_state:
                st.session_state['retrieval_document_chain_memory'] = retrieval_document_chain_memory

            if 'store_history' not in st.session_state:
                st.session_state['store_history'] = retrieval_document_chain_memory.store_history

            st.success("Initialization complete. You can start chatting!")

    # End Chat and Update Journals and Histories
    if "chat_active" in st.session_state and st.session_state['chat_active'] and st.button("End Chat"):
        store_history = st.session_state['retrieval_document_chain_memory'].store_history
        llm = st.session_state['retrieval_document_chain_memory'].llm
        end_chat(user_directory, user_id, session_id, store_history, llm)

    # Chat Interface
    if "retrieval_document_chain_memory" in st.session_state and st.button("Start Chat"):
        st.session_state['chat_active'] = True
        st.session_state['messages'] = [{"role": "assistant", "content": "How can I help you?"}]

    if "chat_active" in st.session_state and st.session_state['chat_active']:
        chat_interface()

    


if __name__ == "__main__":
    main()
