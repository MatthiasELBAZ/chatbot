from Self_Coach_It_Classes import *


def main():

    user_directory = input('user data folder: ')
    if user_directory[-1] != '/':
        user_directory = user_directory + '/'
    user_id = input('user id: ')
    user_id = str(user_id)
    session_id = input('session id: ')
    session_id = str(session_id)
    llm_name_model = input('LLM name model [openai_3.5, openai_4, openai_4o, anthropic_opus, anthropic_sonnet, mistral_large, mistral_8x22B]: ')#"mistral_8x22B"
    embedding_name_model = input('Embedding name model [openai, mistralai]- anthropic not have emdedding model: ')#"openai"
    temperature = input('LLM temperature ([0,1]): ')#0.15
    temperature = float(temperature)
    decay_rate = input('Time Retriever decay rate ([0,1]): ')#1e-5
    decay_rate = float(decay_rate)
    k = input('Time Retriever k (odd integer): ')#7
    k = int(k)
    buffer_num_ongo_messages = input('buffer num ongoing messages (even integer): ')#10
    buffer_num_ongo_messages = int(buffer_num_ongo_messages)
    coach = input('select your coach [career, love, health, finance, general] - pick only career: ')#
    coach = str(coach)
    index_name = input('Pinecone index name: ')
    namespace = input('Pinecone namespace: ')

    # select prompt
    prompts = SelectPrompt('prompt_librairy.json', coach)

    # user definition - existance verification needed
    user_starting_date = "2024-04-10 16:30:20.856632"
    User_Session = UserSessionStoreHistory(user_id, session_id, user_starting_date)
    User_Session.initialize_store_history()
    store_history = User_Session.store_history
    print('User Identity')
    print(f"User ID: {User_Session.user_id}")
    print(f"Session ID: {User_Session.session_id}")
    print(f"Starting Date: {User_Session.user_starting_date}")
    print()
    print('User Store History')
    print(f"Store History: {store_history}")
    print()

    # user directory loader
    User_Loader = UserDirectoryLoader(user_directory)
    user_starting_date = User_Session.user_starting_date
    formulaire_docs_intemp = User_Loader.load_csv_from_directory('formulaire.csv', None)
    formulaire_docs = User_Loader.load_csv_from_directory('formulaire.csv', user_starting_date)
    journal_docs = User_Loader.load_csv_from_directory('journal.csv', user_starting_date)
    history_docs = User_Loader.load_csv_from_directory('history.csv', user_starting_date)
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
    if llm_name_model == 'openai_4o':
        llm_api_key = os.getenv("OPENAI_API_KEY")
    if llm_name_model == 'anthropic_opus':
        llm_api_key = os.getenv("ANTHROPIC_API_KEY")
    if llm_name_model == 'anthropic_sonnet':
        llm_api_key = os.getenv("ANTHROPIC_API_KEY")
    if llm_name_model == 'mistral_large':
        llm_api_key = os.getenv("MISTRALAI_API_KEY")
    if llm_name_model == 'mistral_8x22B':
        llm_api_key = os.getenv("MISTRALAI_API_KEY")

    if embedding_name_model == 'openai':
        embedding_api_key = os.getenv("OPENAI_API_KEY")
    if embedding_name_model == 'mistralai':
        embedding_api_key = os.getenv("MISTRALAI_API_KEY")

    Load_Models = LoadModels(llm_api_key, embedding_api_key, llm_name_model, embedding_name_model, temperature=0.15, max_tokens=4096)
    llm = Load_Models.select_llm_model()
    embedding = Load_Models.select_embedding_model()
    print('Models Loaded')
    print()

    # user profile
    User_Profile = UserProfile(llm, formulaire_docs, journal_docs, Prompts)
    User_Profile.user_profile_generation()
    user_profile = User_Profile.user_profile
    print('User Profile')
    print(f"User Profile: {user_profile}")
    print()

    # initialize pinecone time weighted retriever
    Pinecone_Time_Retiever = PineconeTimeWeightedRetriever(embedding, decay_rate=decay_rate, k=k)
    Pinecone_Time_Retiever.get_pinecone_index(index_name)
    Pinecone_Time_Retiever.get_vectorstore(index_name, namespace)
    Pinecone_Time_Retiever.get_time_retriever()

    # get Pincecone Index stats
    r = Pinecone_Time_Retiever.print_index_stats()
    print(r)

    # check if time retriever is not empty then add documents into memory stream
    if len(r.namespaces)>0:
        print('feed with pinecone index')
        Pinecone_Time_Retiever.time_retriever_add_from_index(namespace)

    # check if time retriever is empty from any backstory
    if len(r.namespaces)==0 and len(journal_docs)>0:
        print('feed with journal backstory')
        Pinecone_Time_Retiever.time_retriever_add_from_documents(journal_docs)

        # get Pincecone Index stats
        time_retriever.get_pinecone_index(index_name)
        r = Pinecone_Time_Retiever.print_index_stats()
        print(r)
    pinecone_time_retriever = Pinecone_Time_Retiever.time_retriever

    # formulaire retriever
    Formulaire_Retriever = ChromaFormulaireRtriever(formulaire_docs_intemp)
    Formulaire_Retriever.get_formulaire_vectorstore()
    Formulaire_Retriever.get_formulaire_retriever()
    formulaire_retriever = Formulaire_Retriever.formulaire_retriever

    # chatbot
    Retrieval_Document_Chain_Memory = RetrievalDocumentChainMemory(
        llm, 
        pinecone_time_retriever,
        formulaire_retriever,
        user_profile, 
        user_id, 
        session_id, 
        store_history, 
        user_starting_date, 
        buffer_num_ongo_messages, 
        Prompts)
    Retrieval_Document_Chain_Memory.get_document_chain()
    Retrieval_Document_Chain_Memory.get_document_chain_with_message_history()
    Retrieval_Document_Chain_Memory.get_retrieval_document_chain_with_message_history()
    print('Retrieval Document Chain Memory')
    print()

    # run chat
    print("Chatbot is ready. Type 'exit' to quit.")
    final_price = 0
    while True:
        promt = input("You: ")
        if promt.lower() == 'exit':
            # get new journal for full messages
            New_Journal = NewJournal(llm, user_id, session_id, store_history, prompts)
            New_Journal.get_new_journal()
            new_journal = New_Journal.new_journal
            print(f"new journal: {new_journal}")
            print()

            # update store
            Update_Store = UpdateStore(
                user_directory+'journal.csv', 
                user_directory+'history.csv', 
                new_journal, 
                store_history, 
                user_id, 
                session_id)
            Update_Store.update_store_journal_csv()
            Update_Store.update_store_chat_history()
            print("Store updated.")
            print()
            #update pinecone index
            metadata = {
                "created_at": get_today()
                }
            doc = Document(page_content=new_journal, metadata=metadata)
            Pinecone_Time_Retiever.time_retriever_add_from_documents([doc])
            print("Pinecone index updated.")
            print()
            print(f"final price: {final_price}")
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
