from Self_Coach_It_Classes import *


def main():

    user_directory = input('user data folder: ')

    # user definition - existance verification needed
    user_id = "0"
    session_id = "2"
    user_session = UserSessionStoreHistory(user_id, session_id)
    user_session.initialize_store_history()
    store_history = user_session.store_history
    print('User Identity')
    print(f"User ID: {user_session.user_id}")
    print(f"Session ID: {user_session.session_id}")
    print()
    print('User Store History')
    print(f"Store History: {store_history}")
    print()

    # user directory loader
    user_loader = UserDirectoryLoader(user_directory)
    formulaire_docs = user_loader.load_csv_from_directory('formulaire.csv')
    journal_docs = user_loader.load_csv_from_directory('journal.csv')
    history_docs = user_loader.load_csv_from_directory('history.csv')
    print('User Directory Loader')
    print(f"Length Formulaire Docs: {len(formulaire_docs)}")
    print(f"Length Journal Docs: {len(journal_docs)}")
    print(f"Length History Docs: {len(history_docs)}")
    print()

    # load models
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm_name_model = "openai"
    embedding_name_model = "openai"
    load_models = LoadModels(openai_api_key, llm_name_model, embedding_name_model)
    llm = load_models.select_llm_model()
    embedding = load_models.select_embedding_model()
    print('Load Models')
    print()

    # user profile
    user_profile = UserProfile(llm, formulaire_docs, journal_docs)
    user_profile.user_profile_generation()
    user_profile = user_profile.user_profile
    print('User Profile')
    print(f"User Profile: {user_profile}")
    print()

    # time weighted retriever
    time_retriever = TimeWeightedRetriever(embedding, "faiss")
    time_retriever.get_vectorstore()
    time_retriever.get_time_retriever()
    time_retriever.time_retriever_add_documents(journal_docs)
    faiss_time_retriever = time_retriever.time_retriever
    print('Time Weighted Retriever')
    print()

    # chatbot
    retrieval_document_chain_memory = RetrievalDocumentChainMemory(
        llm, 
        faiss_time_retriever, 
        user_profile, 
        user_id, 
        session_id, 
        store_history)
    retrieval_document_chain_memory.get_document_chain()
    retrieval_document_chain_memory.get_document_chain_with_message_history()
    retrieval_document_chain_memory.get_retrieval_document_chain_with_message_history()
    print('Retrieval Document Chain Memory')
    print()

    # run chat
    print("Chatbot is ready. Type 'exit' to quit.")
    while True:
        promt = input("You: ")
        if promt.lower() == 'exit':
            # get new journal for full messages
            new_journal = NewJournal(llm, user_id, session_id, store_history)
            new_journal.get_new_journal()
            new_journal = new_journal.new_journal
            print(f"new journal: {new_journal}")
            print()

            # update store
            update_store = UpdateStore(
                'georgette_2/journal.csv', 
                'georgette_2/history.csv', 
                new_journal, 
                store_history, 
                user_id, 
                session_id)
            update_store.update_store_journal_csv()
            update_store.update_store_chat_history()
            print("Store updated.")
            print()

            print("Goodbye!")
            break

        # run chat
        response, callback = retrieval_document_chain_memory.run_chat(promt)

        # print chatbot response
        print("Bot:", response['answer'])
        print()
        store_history = retrieval_document_chain_memory.store_history
        print(f"length store history ongo: {len(store_history[user_id][session_id]['ongo'].messages)}")
        print(f"length store history full: {len(store_history[user_id][session_id]['full'].messages)}")
        print()
        print(f"callback: {callback}")
        print()


if __name__ == "__main__":
    main()
