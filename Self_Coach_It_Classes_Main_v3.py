from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import MessagesPlaceholder, PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.callbacks.base import BaseCallbackHandler
from PineconeModifiedTimeWeightedRetriever import Pinecone_Modified_TimeWeightedVectorStoreRetriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_community.docstore import InMemoryDocstore
from langchain_community.callbacks import get_openai_callback
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_core.document_loaders import BaseLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import faiss
import datetime
import pandas as pd
import re
import pprint
from pathlib import Path
import tempfile
import os
import json
from typing import Dict, Any, List
from pinecone import Pinecone, ServerlessSpec
from langsmith import Client



# utility functions
def get_today():
    return datetime.datetime.now()


def read_json_file(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)


class SelectPrompt():
    def __init__(self, json_file: str, coach_name: str) -> None:
        self.json_file = json_file
        self.coach_name = coach_name
        self.prompt_dict = read_json_file(json_file)
        self.system_prompt_coach = self.prompt_dict['system_prompt_coach'][coach_name]
        self.user_prompt_coach = self.prompt_dict['user_prompt_coach'][coach_name]
        self.journal_user_summary_system_prompt = self.prompt_dict['prompt_summary'][coach_name]['journal_user_summary_system_prompt']
        self.user_summary_user_prompt = self.prompt_dict['prompt_summary'][coach_name]['user_summary_user_prompt']
        self.journal_summary_user_prompt = self.prompt_dict['prompt_summary'][coach_name]['journal_summary_user_prompt']
        self.user_profile_system_prompt = self.prompt_dict['prompt_summary'][coach_name]['user_profile_system_prompt']
        self.user_profile_user_prompt = self.prompt_dict['prompt_summary'][coach_name]['user_profile_user_prompt']
        self.ongo_summary_user_prompt = self.prompt_dict['prompt_summary'][coach_name]['ongo_summary_user_prompt']
        self.full_chat_to_journal_user_prompt = self.prompt_dict['prompt_summary'][coach_name]['full_chat_to_journal_user_prompt']


class UserSessionStoreHistory():
    def __init__(self) -> None:
        self.store_history = {}

    def initialize_store_history(self, user_id: str, session_id: str) -> Dict[str, Any]:
        # initializes the store history for a user session with chat message history
        # self.store_history = {
        #     user_id: {
        #         session_id: {
        #             'ongo': ChatMessageHistory(),
        #             'full': ChatMessageHistory()
        #         }
        #     }
        # }
        self.store_history[(user_id, session_id)] = {
            'ongo': ChatMessageHistory(),
            'full': ChatMessageHistory()
        }


class UserDirectoryLoader():
    def __init__(self, directory: str) -> None:
        self.directory = directory
    
    def add_date_to_documents(self, docs: List, user_starting_date: str) -> List:
        for doc in docs:
            # get data
            page_content = doc.page_content
            metadata = doc.metadata

            # the date appears in the page content from date column in csv alias DB
            match = re.search(r'date: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6})', page_content)
            if match:
                # modification metadata to add the date from the page content - time retriever related
                metadata["created_at"] = datetime.datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S.%f')
            else:
                # edge case for formulaire - date the formulaire with the user starting date
                metadata["created_at"] = datetime.datetime.strptime(user_starting_date, '%Y-%m-%d %H:%M:%S.%f')
                
        return docs

    def load_csv_from_directory(self, csv_file_name: str, user_starting_date: str) -> List:
        loader = DirectoryLoader(self.directory, glob=csv_file_name, loader_cls=CSVLoader)
        docs = loader.load()

        # tigger the add_date_to_documents function with the user starting date
        if user_starting_date:
            docs = self.add_date_to_documents(docs, user_starting_date) 

        return docs


class ChromaFormulaireRtriever():
    def __init__(self) -> None:
        self.formulaire_vectorstore = None
        self.formulaire_retriever = None
    
    def format_docs(self, docs: List) -> str:
        return "The user said about himself in the form:\n\n"+"\n\n".join(doc.page_content.split('Answers: ')[-1] for doc in docs)

    def get_formulaire_vectorstore(self, formulaire_docs: List, embedding: Any) -> None:
        self.formulaire_vectorstore = Chroma.from_documents(documents=formulaire_docs, embedding=embedding)
    
    def get_formulaire_retriever(self) -> None:
        self.formulaire_retriever = self.formulaire_vectorstore.as_retriever()
        self.formulaire_retriever = self.formulaire_retriever | self.format_docs


class PineconeTimeWeightedRetriever():
    def __init__(self) -> None:
        self.vectorstore = None
        self.time_retriever = None
        self.time_retriever_chain = None
        self.pinecone_index = None

    def format_docs(self, docs: List) -> str:
        return "Information extracted from Journal:\n\n"+"\n\n".join(doc.page_content for doc in docs)
        
    def get_vectorstore(self, index_name: str, namespace: str, embedding: Any, pinecone_api_key: str) -> None:
        # declare vectore store
        vectorstore = PineconeVectorStore(
            index_name=index_name, 
            embedding=embedding, 
            namespace=namespace, 
            pinecone_api_key=pinecone_api_key
            )
        # assign vectorstore to self
        self.vectorstore = vectorstore
        
    def get_time_retriever(self, decay_rate: float, k: int) -> None:
        # declare time retriever
        time_retriever = Pinecone_Modified_TimeWeightedVectorStoreRetriever(
            vectorstore=self.vectorstore, 
            decay_rate=decay_rate, 
            k=k
            )
        # assign time retriever to self
        self.time_retriever = time_retriever

    def get_pinecone_index(self, index_name: str, pinecone_api_key: str) -> None:
        pc = Pinecone(api_key=pinecone_api_key)
        self.pinecone_index = pc.Index(index_name)

    def time_retriever_add_from_index(self, namespace: str) -> None:
        memory_stream = []
        for id in list(self.pinecone_index.list(namespace=namespace))[0]:
            # fetch doc from pinecone index
            doc = self.pinecone_index.fetch([id],namespace=namespace)
            page_content = doc['vectors'][id]['metadata']['text']
            metadata = doc['vectors'][id]['metadata']

            # modification of the metadata to have the right format
            metadata['created_at'] = datetime.datetime.strptime(metadata['created_at'], '%Y-%m-%dT%H:%M:%S.%f')
            metadata['last_accessed_at'] = datetime.datetime.strptime(metadata['last_accessed_at'], '%Y-%m-%dT%H:%M:%S.%f')
            metadata['buffer_idx'] = int(metadata['buffer_idx'])
            metadata.pop('text', None)

            # append to memory stream
            memory_stream.append(Document(page_content=page_content, metadata=metadata))
        
        # assign memory stream to time retriever
        self.time_retriever.memory_stream = memory_stream

    def print_index_stats(self) -> Any:
        return self.pinecone_index.describe_index_stats()

    def delete_namespace(self, namespace: str) -> None:
        self.pinecone_index.delete(namespace=namespace,  delete_all=True)

    def make_it_chain(self):
        self.time_retriever_chain = self.time_retriever | self.format_docs


class UserProfile():
    def __init__(self) -> None:
        self.summarizer = None
        self.user_summary = None
        self.journal_summary = None
        self.user_profile = None

    def make_summarizer(self, llm: Any ,prompts: Dict) -> None:
        # define summarizer prompt
        summary_prompt = ChatPromptTemplate.from_messages([("system", prompts.journal_user_summary_system_prompt), MessagesPlaceholder(variable_name="messages")])
        # define summarizer chain
        summarizer = create_stuff_documents_chain(llm, summary_prompt)
        # assign summarizer to self
        self.summarizer = summarizer

    def get_user_summary(self, formulaire_docs: List, prompts: Dict) -> None:
        # invoke summarizer
        user_summary = self.summarizer.invoke({"context": formulaire_docs, "messages": [HumanMessage(content=prompts.user_summary_user_prompt)]})
        # assign user summary to self
        self.user_summary = user_summary
        
    def get_journal_summary(self, journal_docs: List, prompts: Dict) -> None:
        # invoke summarizer
        journal_summary = self.summarizer.invoke({"context": journal_docs, "messages": [HumanMessage(content=prompts.journal_summary_user_prompt)]})
        # assign journal summary to self
        self.journal_summary = journal_summary

    def get_user_profile(self, llm: Any, user_summary: str, journal_summary: str, prompts: Dict) -> None:
        # define user profiler prompt
        user_profiler_prompt = ChatPromptTemplate.from_messages([("system", prompts.user_profile_system_prompt), MessagesPlaceholder(variable_name="messages")])
        # define user profiler chain
        user_profiler_chain = user_profiler_prompt | llm
        # invoke user profiler
        user_profile = user_profiler_chain.invoke(
            {
                "user_summary": user_summary, 
                "journal_summary": journal_summary,
                "messages": [HumanMessage(content=prompts.user_profile_user_prompt)]
            }).content
        # assign user profile to self
        self.user_profile = user_profile
        
    def user_profile_generation(self, llm: Any, prompts: Dict, formulaire_docs: List,  journal_docs: List) -> None:
        # make summarizer
        self.make_summarizer(llm, prompts)
        # get user summary
        self.get_user_summary(formulaire_docs, prompts)
        if len(journal_docs) == 0:
            self.user_profile = self.user_summary
        else:
            # get journal summary
            self.get_journal_summary(journal_docs, prompts)
            self.get_user_profile(llm, self.user_summary, self.journal_summary, prompts)


class CallBacker(BaseCallbackHandler):
    def __init__(self, llm):
        self.llm = llm
        # get model name
        try:
            self.llm_name_model = llm.model
        except:
            self.llm_name_model = llm.model_name
        
        # get price per token
        self.price_input = 5/1e6
        self.price_output = 15/1e6
    
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        for p in prompts:
            self.input_tokens += self.llm.get_num_tokens(p)

    def on_llm_end(self, response, **kwargs):
        results = response.flatten()
        for r in results:
            self.output_tokens = self.llm.get_num_tokens(r.generations[0][0].text)

    def get_request_price(self):
        return self.input_tokens*self.price_input + self.output_tokens*self.price_output 


class RetrievalDocumentChainMemory():
    def __init__(self) -> None:
        self.retrieval_document_chain_with_message_history = None
        self.ongo_summary_message = None
    
    def parse_retriever_input(self, params: Dict) -> str:
        # return input from user
        print(params)
        return params["input"]

    def get_session_history(self, user_id: str, session_id: str, store_history: Dict) -> BaseChatMessageHistory:
        # Retrieve the chat message history for a given session, initializing if necessary.
        # return store_history.get(user_id, {}).get(session_id, {
        #     'ongo': ChatMessageHistory(), 
        #     'full': ChatMessageHistory()
        # })['ongo']
        return store_history[(user_id,session_id)]['ongo']

    def summarize_ongo_messages(self, chain_input: Any, llm: Any, prompts: Dict, user_id: str, session_id: str, store_history: Dict, buffer_num_ongo_messages: int) -> bool:
        # get last messages - last conversation + summary until last conversation
        stored_messages = store_history[(user_id,session_id)]['ongo'].messages
        
        # do nothing if no messages
        if len(stored_messages)<buffer_num_ongo_messages:
            return False

        else:
            # summarization prompt
            summarization_prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="chat_history"), ("user", prompts.ongo_summary_user_prompt)])
            # summarization chain
            summarization_chain = summarization_prompt | llm
            # invoke summarization chain
            ongo_summary_message = summarization_chain.invoke({"chat_history": stored_messages})
            # assign summary message
            self.ongo_summary_message = ongo_summary_message
            return True
    
    def get_retrieval_document_chain_with_message_history(self, llm: Any, prompts: Dict, user_id: str, session_id: str, store_history: Dict, buffer_num_ongo_messages: int) -> None:
        # define chatbot prompt chain
        coach_chatbot_prompt = ChatPromptTemplate.from_messages([("system",prompts.system_prompt_coach), MessagesPlaceholder("chat_history"), ("human", prompts.user_prompt_coach)])
        document_chain = coach_chatbot_prompt | llm

        # add it memoryget_session_history
        document_chain_with_message_history = RunnableWithMessageHistory(
            document_chain,
            get_session_history=lambda get_session_history: self.get_session_history(user_id, session_id, store_history),
            input_messages_key="input",
            history_messages_key="chat_history",
            answer_messages_key="answer"
        )
        # assign chat memory summarization and assign answer
        retrieval_document_chain_with_message_history = (
            RunnablePassthrough.assign(messages_summarized=lambda chain_input: self.summarize_ongo_messages(chain_input ,llm, prompts, user_id, session_id, store_history, buffer_num_ongo_messages)).assign(answer=document_chain_with_message_history)
            )
        # assign document chain to self
        self.retrieval_document_chain_with_message_history = retrieval_document_chain_with_message_history

    def run_chat(self, input: str, user_profile: str, time_retriever: Any, formulaire_retriever: Any, user_id: str, session_id: str, store_history: Dict, callbacker: Any) -> [Dict, Any]:

        # run chat with callback
        result = self.retrieval_document_chain_with_message_history.invoke(
            {
                'date_today': get_today(),
                'user_profile': user_profile,
                "rag_journal_context": time_retriever.invoke(input),
                "rag_user_info_context": formulaire_retriever.invoke(input),
                "input":input
            },
            config={
                "configurable": {"user_id": user_id, "session_id": session_id}, 
                "callbacks": [callbacker]
            }
        )

        # manage store history
        if result['messages_summarized']==True:
            # clear ongoing messages and add summary message
            store_history[(user_id,session_id)]['ongo'].clear()

            # add last buffer_num_ongo_messages messages to ongoing messages 
            # for message in store_history[user_id][session_id]['full'].messages[-buffer_num_ongo_messages:]:
            #     store_history[user_id][session_id]['ongo'].add_message(message)

            # add human message before summary message for anthropic model
            store_history[(user_id,session_id)]['ongo'].add_message(HumanMessage("Make a summary of our chat messages unitl now."))

            # add summary message to ongoing messages
            store_history[(user_id,session_id)]['ongo'].add_message(self.ongo_summary_message)

        # fill store history full with the input user and chatbot messages
        store_history[(user_id,session_id)]['full'].add_message(HumanMessage(input))
        store_history[(user_id,session_id)]['full'].add_message(AIMessage(result['answer'].content))

        return result, {'input_tokens':callbacker.input_tokens, 'output_tokens':callbacker.output_tokens, 'price':callbacker.get_request_price()}


class NewJournal():
    def __init__(self) -> None:
        self.new_journal = None

    def get_new_journal(self, llm: Any, prompts: Any, user_id: str, session_id: str, store_history: Dict) -> None:
        # get full messages from sate session
        full_messages = store_history[user_id][session_id]['full'].messages

        # define journal summarizer prompt
        summarization_prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="chat_history"), ("user", prompts.full_chat_to_journal_user_prompt)])

        # instance the summarization chain
        summarization_chain = summarization_prompt | llm

        # invoke the chain
        new_journal = summarization_chain.invoke({"chat_history": full_messages})

        # assign document chain to self
        self.new_journal = new_journal.content


class UpdateStore():
    def __init__(self):
        return None

    def update_store_journal_csv(self, journal_csv_path: str, new_journal: str, session_id: str) -> None:
        # load pandas journal.csv
        df = pd.read_csv(journal_csv_path)

        # add new line at date of today with the summary message
        r = pd.DataFrame({str(len(df)+1):{'date': get_today(), 'sentence': new_journal}}).T
        r['session_id'] = session_id 
        df = pd.concat([df, r], ignore_index=True)

        # save the new dataframe
        df.to_csv(self.journal_csv_path, index=False)

    def update_journal_pinecone_index(self, Pinecone_Time_Retiever: Any, new_journal: str, namespace: str, coach: str) -> None:
        # add new journal to pinecone index
        metadata = {
                "created_at": get_today(),
                "coach": coach
            }
        doc = Document(page_content=new_journal, metadata=metadata)
        Pinecone_Time_Retiever.time_retriever_add_from_documents([doc])

    def update_store_chat_history_csv(self, chat_history_csv_path: str, store_history: Dict, user_id: str, session_id: str) -> None:
         # new chat history
        new_df_chat_history = pd.DataFrame(store_history[user_id][session_id]['full'].dict()['messages'])
        new_df_chat_history['date'] = get_today()
        new_df_chat_history = new_df_chat_history[['date', 'content', 'type']]
        new_df_chat_history['sesion_id'] = session_id

        # old chat history
        df_chat_history = pd.read_csv(chat_history_csv_path, index_col=0)

        # add new_df_chat_messages to df_chat_messages
        df_chat_history = pd.concat([df_chat_history, new_df_chat_history], ignore_index=True).reset_index(drop=True)

        # save the new dataframe
        df_chat_history.to_csv(chat_history_csv_path, index=False)



def initialization(user_directory, user_id, session_id, user_starting_date, namespace, index_name, coach_name, temperature, decay_rate, k, buffer_num_ongo_messages):

    # select prompt from json file path
    prompts = SelectPrompt('prompt_librairy.json', coach_name)

    # store history initialization
    User_Session = UserSessionStoreHistory()
    User_Session.initialize_store_history(user_id, session_id)
    store_history = User_Session.store_history

    # user directory loader
    User_Loader = UserDirectoryLoader(user_directory)
    formulaire_docs = User_Loader.load_csv_from_directory('formulaire.csv', None)

    # initialize model and embedding
    llm_api_key = os.getenv("OPENAI_API_KEY")
    embedding_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o", temperature=temperature, max_tokens=4096, openai_api_key=llm_api_key)
    embedding = OpenAIEmbeddings(openai_api_key=embedding_api_key)

    # Formulaire retrieval
    Chroma_Formulaire = ChromaFormulaireRtriever()
    Chroma_Formulaire.get_formulaire_vectorstore(formulaire_docs, embedding)
    Chroma_Formulaire.get_formulaire_retriever()

    # Pinecone Time Weighted Retriever
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    Pinecone_Time_Retiever_Journal = PineconeTimeWeightedRetriever()
    Pinecone_Time_Retiever_Journal.get_pinecone_index(index_name, pinecone_api_key)
    Pinecone_Time_Retiever_Journal.get_vectorstore(index_name, namespace, embedding, pinecone_api_key)
    Pinecone_Time_Retiever_Journal.get_time_retriever(decay_rate, k)
    Pinecone_Time_Retiever_Journal.time_retriever_add_from_index(namespace)
    Pinecone_Time_Retiever_Journal.make_it_chain()

    # get jornal docs from memory stream
    journal_docs = Pinecone_Time_Retiever_Journal.time_retriever.memory_stream

    # User Profile
    User_Profile = UserProfile()
    User_Profile.user_profile_generation(llm, prompts, formulaire_docs, journal_docs)

    # get callbacker
    Call_Backer = CallBacker(llm)

    # Retrieval Document Chain Memory
    Retrieval_Document_Chain_Memory = RetrievalDocumentChainMemory()
    Retrieval_Document_Chain_Memory.get_retrieval_document_chain_with_message_history(llm, prompts, user_id, session_id, store_history, buffer_num_ongo_messages)

    # store all in a dictionary
    initialization_dict = {
        'user_directory': user_directory,
        'user_id': user_id,
        'session_id': session_id,
        'user_starting_date': user_starting_date,
        'namespace': namespace,
        'index_name': index_name,
        'coach_name': coach_name,
        'temperature': temperature,
        'decay_rate': decay_rate,
        'k': k,
        'buffer_num_ongo_messages': buffer_num_ongo_messages,
        'prompts': prompts,
        'store_history': store_history,
        'llm': llm,
        'embedding': embedding,
        'Chroma_Formulaire': Chroma_Formulaire,
        'Pinecone_Time_Retiever_Journal': Pinecone_Time_Retiever_Journal,
        'User_Profile': User_Profile,
        'Call_Backer':Call_Backer,
        'Retrieval_Document_Chain_Memory': Retrieval_Document_Chain_Memory
    }

    return initialization_dict


def run_chat(input, initialization_dict):
    # get all the necessary variables
    user_directory = initialization_dict['user_directory']
    user_id = initialization_dict['user_id']
    session_id = initialization_dict['session_id']
    user_starting_date = initialization_dict['user_starting_date']
    namespace = initialization_dict['namespace']
    index_name = initialization_dict['index_name']
    coach_name = initialization_dict['coach_name']
    temperature = initialization_dict['temperature']
    decay_rate = initialization_dict['decay_rate']
    k = initialization_dict['k']
    buffer_num_ongo_messages = initialization_dict['buffer_num_ongo_messages']
    prompts = initialization_dict['prompts']
    store_history = initialization_dict['store_history']
    llm = initialization_dict['llm']
    embedding = initialization_dict['embedding']
    Chroma_Formulaire = initialization_dict['Chroma_Formulaire']
    Pinecone_Time_Retiever_Journal = initialization_dict['Pinecone_Time_Retiever_Journal']
    User_Profile = initialization_dict['User_Profile']
    Call_Backer = initialization_dict['Call_Backer']
    Retrieval_Document_Chain_Memory = initialization_dict['Retrieval_Document_Chain_Memory']

    user_profile = User_Profile.user_profile
    formulaire_retriever = Chroma_Formulaire.formulaire_retriever
    journal_time_retriever_chain = Pinecone_Time_Retiever_Journal.time_retriever_chain

    # run chat
    if input.lower() == 'exit':
        results = None
        callback = None

        # get new journal for full messages
        New_Journal = NewJournal()
        New_Journal.get_new_journal(llm, prompts, user_id, session_id, store_history)
        new_journal = New_Journal.new_journal
        print("New Journal made.")
        print()
        # update store
        Update_Store = UpdateStore()
        Update_Store.update_store_journal_csv()
        Update_Store.update_store_chat_history_csv()
        print("CSV Store updated.")
        print()
        #update pinecone index
        Update_Store.update_journal_pinecone_index(Pinecone_Time_Retiever_Journal, new_journal, namespace, coach_name)
        print("Pinecone index updated.")
        print()
        print("Goodbye!")

    else:
        result, callback = Retrieval_Document_Chain_Memory.run_chat(input, user_profile, journal_time_retriever_chain, formulaire_retriever, user_id, session_id, store_history, Call_Backer)

    # store all in a dictionary
    chat_dict = {
        'user_directory': user_directory,
        'user_id': user_id,
        'session_id': session_id,
        'user_starting_date': user_starting_date,
        'namespace': namespace,
        'index_name': index_name,
        'coach_name': coach_name,
        'temperature': temperature,
        'decay_rate': decay_rate,
        'k': k,
        'buffer_num_ongo_messages': buffer_num_ongo_messages,
        'prompts': prompts,
        'store_history': store_history,
        'llm': llm,
        'embedding': embedding,
        'Chroma_Formulaire': Chroma_Formulaire,
        'Pinecone_Time_Retiever_Journal': Pinecone_Time_Retiever_Journal,
        'User_Profile': User_Profile,
        'Retrieval_Document_Chain_Memory': Retrieval_Document_Chain_Memory,
        'result': result,
        'callback': callback
    }

    return chat_dict


def main():

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "conversation_1"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = "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"

    client = Client()

    user_directory = 'georgette_2/'
    user_id = '0'
    session_id = '17'
    temperature = 0.05
    decay_rate = 0.0005
    k = 3
    buffer_num_ongo_messages = 4
    coach = 'career'
    index_name = 'matthiasdb'
    namespace = 'georgette'
    user_starting_date = "2024-04-10 16:30:20.856632"

    # initialization
    initialization_dict = initialization(user_directory, user_id, session_id, user_starting_date, namespace, index_name, coach, temperature, decay_rate, k, buffer_num_ongo_messages)

    # run chat
    print("Chatbot is ready. Type 'exit' to quit.")
    final_price = 0
    while True:
        prompt = input('You: ')
        chat_dict = run_chat(prompt, initialization_dict)
        response = chat_dict['result']
        callback = chat_dict['callback']
        store_history = chat_dict['store_history']
        if response is None:
            print(f"final price: {final_price}")
            break
        else:
            print("Bot:", response['answer'].content)
            print()
            print(f"length store history ongo: {len(store_history[(user_id,session_id)]['ongo'].messages)}")
            print(f"length store history full: {len(store_history[(user_id,session_id)]['full'].messages)}")
            print()
            print(f"callback: {callback}")
            print()
            final_price += callback['price']


if __name__ == "__main__":
    main()