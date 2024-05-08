from Self_Coach_It_Classes import *
import pandas as pd



def main():

    user_directory = input('user data folder: ')
    if user_directory[-1] != '/':
        user_directory = user_directory + '/'
    user_id = input('user id: ')
    user_id = str(user_id)
    session_id = input('session id: ')
    session_id = str(session_id)
    llm_name_model = input('LLM name model [openai_3.5, openai_4, anthropic_opus, anthropic_sonnet, mistral_large, mistral_8x22B]: ')#"mistral_8x22B"
    embedding_name_model = input('Embedding name model [openai_3.5, openai_4, mistral_large, mistral_8x22B]- anthropic not have emdedding model: ')#"openai_3.5"
    temperature = input('LLM temperature ([0,1]): ')#0.15
    temperature = float(temperature)
    decay_rate = input('Time Retriever decay rate ([0,1]): ')#1e-5
    decay_rate = float(decay_rate)
    k = input('Time Retriever k (odd integer): ')#7
    k = int(k)
    buffer_num_ongo_messages = input('buffer num ongoing messages (even integer): ')#10
    buffer_num_ongo_messages = int(buffer_num_ongo_messages)

    # user definition - existance verification needed
    user_starting_date = "2024-04-10 16:30:20.856632"
    user_session = UserSessionStoreHistory(user_id, session_id, user_starting_date)
    user_session.initialize_store_history()
    store_history = user_session.store_history
    print('User Identity')
    print(f"User ID: {user_session.user_id}")
    print(f"Session ID: {user_session.session_id}")
    print(f"Starting Date: {user_session.user_starting_date}")
    print()
    print('User Store History')
    print(f"Store History: {store_history}")
    print()

    # user directory loader
    user_loader = UserDirectoryLoader(user_directory)
    user_starting_date = user_session.user_starting_date
    formulaire_docs = user_loader.load_csv_from_directory('formulaire.csv', user_starting_date)
    journal_docs = user_loader.load_csv_from_directory('journal.csv', user_starting_date)
    history_docs = user_loader.load_csv_from_directory('history.csv', user_starting_date)
    print('User Directory Loader')
    print(f"Length Formulaire Docs: {len(formulaire_docs)}")
    print(f"Length Journal Docs: {len(journal_docs)}")
    print(f"Length History Docs: {len(history_docs)}")
    print()

    # load models
    if llm_name_model == 'openai_3.5':
        llm_api_key = os.getenv("OPENAI_API_KEY")
    if llm_name_model == 'openai_4':
        llm_api_key = os.getenv("OPENAI_API_KEY")
    if llm_name_model == 'anthropic_opus':
        llm_api_key = os.getenv("ANTHROPIC_API_KEY")
    if llm_name_model == 'anthropic_sonnet':
        llm_api_key = os.getenv("ANTHROPIC_API_KEY")
    if llm_name_model == 'mistral_large':
        llm_api_key = os.getenv("MISTRALAI_API_KEY")
    if llm_name_model == 'mistral_8x22B':
        llm_api_key = os.getenv("MISTRALAI_API_KEY")

    if embedding_name_model == 'openai_3.5':
        embedding_api_key = os.getenv("OPENAI_API_KEY")
    if embedding_name_model == 'openai_4':
        embedding_api_key = os.getenv("OPENAI_API_KEY")
    if embedding_name_model == 'anthropic_opus':
        embedding_api_key = os.getenv("ANTHROPIC_API_KEY")
    if embedding_name_model == 'anthropic_sonnet':
        embedding_api_key = os.getenv("ANTHROPIC_API_KEY")
    if embedding_name_model == 'mistral_large':
        embedding_api_key = os.getenv("MISTRALAI_API_KEY")
    if embedding_name_model == 'mistral_8x22B':
        embedding_api_key = os.getenv("MISTRALAI_API_KEY")

    load_models = LoadModels(llm_api_key, embedding_api_key, llm_name_model, embedding_name_model, temperature=0.15, max_tokens=4096)
    llm = load_models.select_llm_model()
    embedding = load_models.select_embedding_model()
    print('Models Loaded')
    print()

    # user profile
    user_profile = UserProfile(llm, formulaire_docs, journal_docs)
    user_profile.user_profile_generation()
    user_profile = user_profile.user_profile
    print('User Profile')
    print(f"User Profile: {user_profile}")
    print()

    # time weighted retriever
    time_retriever = TimeWeightedRetriever(embedding, "faiss", decay_rate=decay_rate, k=k)
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
        store_history, 
        user_starting_date, 
        buffer_num_ongo_messages=buffer_num_ongo_messages)
    retrieval_document_chain_memory.get_document_chain()
    retrieval_document_chain_memory.get_document_chain_with_message_history()
    retrieval_document_chain_memory.get_retrieval_document_chain_with_message_history()
    print('Retrieval Document Chain Memory')
    print()

    # run chat
    print("Chatbot is ready. Type 'exit' to quit.")
    final_price = 0
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
                user_directory+'journal.csv', 
                user_directory+'history.csv', 
                new_journal, 
                store_history, 
                user_id, 
                session_id)
            update_store.update_store_journal_csv()
            update_store.update_store_chat_history()
            print("Store updated.")
            print()
            print(f"final price of conversation: {final_price}")
            print("Goodbye!")
            break

        # run chat
        response, callback = retrieval_document_chain_memory.run_chat(promt)
        final_price += callback['price']
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
