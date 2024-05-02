from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import MessagesPlaceholder, PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from PineconeModifiedTimeWeightedRetriever import Pinecone_Modified_TimeWeightedVectorStoreRetriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.vectorstores import FAISS
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
import faiss
import datetime
import pandas as pd
import re
import pprint
import streamlit as st
import datetime
from pathlib import Path
import tempfile
import os
from typing import Dict, Any

# utility functions
def get_today():
    return datetime.datetime.now()

class UserSessionIdentity():
    def __init__(self, user_id: str, session_id: str) -> None:
        self._user_id = user_id
        self._session_id = session_id

    @property
    def user_id(self) -> str:
        """Return the user ID."""
        return self._user_id

    @property
    def session_id(self) -> str:
        """Return the session ID."""
        return self._session_id


class UserSessionStoreHistory(UserSessionIdentity):
    def __init__(self, user_id: str, session_id: str) -> None:
        super().__init__(user_id, session_id)
        self.store_history = {}

    def initialize_store_history(self) -> Dict[str, Any]:
        # initializes the store history for a user session with chat message history
        self.store_history = {
            self.user_id: {
                self.session_id: {
                    'ongo': ChatMessageHistory(),
                    'full': ChatMessageHistory()
                }
            }
        }


class UserDirectoryLoader():
    def __init__(self, directory: str) -> None:
        self.directory = directory
    
    def add_date_to_documents(self, docs: list) -> list:
        for doc in docs:
            page_content = doc.page_content
            metadata = doc.metadata
            try:
                # modification metadata to add the date from the page content - time retriever related
                date = re.search(r'\d{2}/\d{2}/\d{4}', page_content).group()
                metadata["created_at"] = datetime.datetime.strptime(date, '%m/%d/%Y')
            except:
                continue
        return docs

    def load_csv_from_directory(self, csv_file_name: str) -> list:
        loader = DirectoryLoader(self.directory, glob=csv_file_name, loader_cls=CSVLoader)
        docs = loader.load()
        docs = self.add_date_to_documents(docs)    
        return docs


class LoadModels():
    def __init__(self, api_key: str, llm_name_model: str, embedding_name_model: str) -> None:
        self.llm_name_model = llm_name_model
        self.embedding_name_model = embedding_name_model
        self.api_key = api_key

    def select_llm_model(self) -> Any:
        if self.llm_name_model == "openai":
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, openai_api_key=self.api_key)
        if self.llm_name_model == "anthropic":
            llm = ChatAnthropic(model='claude-3-opus-20240229', anthropic_api_key=self.api_key)
        if self.llm_name_model == "mistralai":
            llm = ChatMistralAI(model='open-mistral-7b', mistral_api_key=self.api_key)
        return llm

    def select_embedding_model(self) -> Any:
        if self.embedding_name_model == "openai":
            embedding = OpenAIEmbeddings(openai_api_key=self.api_key)
        if self.embedding_name_model == "anthropic":
            embedding = AnthropicEmbeddings()
        if self.embedding_name_model == "mistralai":
            embedding = MistralAIEmbeddings(api_key=self.api_key)
        return embedding


class UserProfile():
    def __init__(self, llm: Any, formulaire_docs: list, journal_docs: list) -> None:
        self.llm = llm
        self.formulaire_docs = formulaire_docs
        self.journal_docs = journal_docs
        self.summarizer = None
        self.user_summary = None
        self.journal_summary = None
        self.user_profile = None

    def make_summarizer(self) -> None:
        # define summarizer prompt
        summary_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are summarizer assitant focusing on job and career actions and thoughts.
                
                You summarizes the user's information, based on the below context and career concerns. 

                Do NOT exceed 300 words.

                If context is empty, return an empty string.

                <context>
                {context}
                </context>
                """
            ), 
            MessagesPlaceholder(variable_name="messages"),
        ]

        )

        # define summarizer chain
        summarizer = create_stuff_documents_chain(self.llm, summary_prompt)

        # assign summarizer to self
        self.summarizer = summarizer

    def get_user_summary(self) -> None:
        # invoke summarizer
        user_summary = self.summarizer.invoke(
            {
                "context": self.formulaire_docs,
                "messages": [
                    HumanMessage(content="Make a summary text of the user's answers to the formulaire. Starts with name and date of birth. do not exeed 300 words.")
                ],
            }
        )

        # assign user summary to self
        self.user_summary = user_summary
        
    def get_journal_summary(self) -> None:
        # invoke summarizer
        journal_summary = self.summarizer.invoke(
            {
                "context": self.journal_docs,
                "messages": [
                    HumanMessage(content="Make a summary text of your observations about the user. do not exeed 300 words.")
                ],
            }
        )

        # assign journal summary to self
        self.journal_summary = journal_summary

    def get_user_profile(self, user_summary: str, journal_summary: str) -> None:
        # define user profiler prompt
        user_profiler_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are an assistant that updates the user summary based on journal summary.

                    Journal summary represents the summary of the previous conversations.

                    It always starts with the user's name and date of birth.

                    Journal focuses on actions and thoughts relative to job and career personal development.

                    Modify the user summary accroding to the journal information to write a new user summary.

                    The new user summary must focus on job and career actions and thoughts.

                    You summarizes the user's information, based on the below context and career concerns.
                
                    Do not exceed 300 words.

                    User Summary:
                    {user_summary}

                    Journal Summary:
                    {journal_summary}

                    """,
                ),
            ]
        )

        # define user profiler chain
        user_profiler_chain = user_profiler_prompt | self.llm

        # invoke user profiler
        user_profile = user_profiler_chain.invoke(
            {
                "user_summary": user_summary, 
                "journal_summary": journal_summary
            }
            ).content

        # assign user profile to self
        self.user_profile = user_profile
        
    def user_profile_generation(self) -> None:
        # make summarizer
        self.make_summarizer()

        # get user summary
        self.get_user_summary()

        if len(self.journal_docs) == 0:
            self.user_profile = self.user_summary
        else:
            # get journal summary
            self.get_journal_summary()
            self.get_user_profile(self.user_summary, self.journal_summary)


class TimeWeightedRetriever():
    def __init__(self, embedding: Any, type: str) -> None:
        self.embedding = embedding
        self.type = type
        self.vectorstore = None
        self.time_retriever = None

    def get_vectorstore(self, index=None) -> None:
        if self.type == "faiss":
            # declare index
            index = faiss.IndexFlatL2(1536)

            # make vectorstore
            vectorstore = FAISS(self.embedding, index, InMemoryDocstore({}), {})
        if self.type == "pinecone":
            # make vectorstore
            vectorstore = PineconeVectorStore(index_name=index, embedding=self.embedding)

        # assign vectorstore to self
        self.vectorstore = vectorstore
        
    def get_time_retriever(self) -> None:
        if self.type == "faiss":
            time_retriever = TimeWeightedVectorStoreRetriever(
                vectorstore=self.vectorstore, 
                decay_rate=1e-5, 
                k=5
            )
        if self.type == "pinecone":
            time_retriever = Pinecone_Modified_TimeWeightedVectorStoreRetriever(
                vectorstore=self.vectorstore, 
                decay_rate=1e-5, 
                k=5
            )

        # assign time retriever to self
        self.time_retriever = time_retriever

    def time_retriever_add_documents(self, docs: list) -> None:
        for i, doc in enumerate(docs):
            page_content = doc.page_content
            metadata = doc.metadata
            self.time_retriever.add_documents([Document(page_content=page_content, metadata=metadata)])


class RetrievalDocumentChainMemory(UserSessionStoreHistory):
    def __init__(self, llm: Any, time_retriever: Any, user_profile: str, user_id: str, session_id: str, store_history: Dict) -> None:
        super().__init__(user_id, session_id)
        self.time_retriever = time_retriever
        self.user_profile = user_profile
        self.store_history = store_history
        self.document_chain = None
        self.document_chain_with_message_history = None
        self.retrieval_document_chain_with_message_history = None
        self.llm = llm

    def parse_retriever_input(self, params: Dict) -> str:
        # return input from user
        return params["input"]
    
    def get_document_chain(self) -> None:
        # define chatbot prompt
        coach_chatbot_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                """
                You are coach for job and career search.

                The user speaks to and you speak to him.

                You talk like Uncle Sam.

                You stay concise and to the point.

                You ask questions to understand the user's career concerns.

                Your guide the user lighting the way to a fulfilling career journey.

                You are capable of jugment on user input related to career search and his profile.

                You tell the user if he is not in the right direction for him.

                The provided chat history summary includes facts about the user you are speaking with.

                this is the date of today conversation: 
                {date_today}

                this is the updated user profile to refer to: 
                {user_profile}

                this is the context to refer to:
                {context}

                DO NOT exceed 50 tokens.

                """
                ),

                MessagesPlaceholder("chat_history"),

                ("human", "{input}"),
            ]
            )

        # define document chain
        document_chain = create_stuff_documents_chain(self.llm, coach_chatbot_prompt)

        # assign document chain to self
        self.document_chain = document_chain

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        # Retrieve the chat message history for a given session, initializing if necessary.
        return self.store_history.get(self.user_id, {}).get(session_id, {
            'ongo': ChatMessageHistory(), 
            'full': ChatMessageHistory()
        })['ongo']
        
    def get_document_chain_with_message_history(self) -> None:
        document_chain_with_message_history = RunnableWithMessageHistory(
            self.document_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            answer_messages_key="answer"
        )

        # assign document chain to self
        self.document_chain_with_message_history = document_chain_with_message_history

    def summarize_ongo_messages(self, chain_input: Any) -> bool:
        # get last messages - last conversation + summary until last conversation
        stored_messages = self.store_history[self.user_id][self.session_id]['ongo'].messages
        
        # do nothing if no messages
        if len(stored_messages)==0:
            return False

        else:
            # summarization prompt
            summarization_prompt = ChatPromptTemplate.from_messages(
                [
                    MessagesPlaceholder(variable_name="chat_history"),
                    (
                        "user",
                        """
                        You are summarizer assitant focusing on job and career actions and thoughts.

                        Summarize the user's chat history based on career concerns. 

                        Do NOT exceed 300 words.
                        
                        """,
                    )
                ]
            )
            # summarization chain
            summarization_chain = summarization_prompt | self.llm

            # invoke summarization chain
            summary_message = summarization_chain.invoke({"chat_history": stored_messages})
            
            # clear ongoing messages and add summary message
            self.store_history[self.user_id][self.session_id]['ongo'].clear()
            self.store_history[self.user_id][self.session_id]['ongo'].add_message(summary_message)
            
            return True

    def get_retrieval_document_chain_with_message_history(self) -> None:
        retrieval_document_chain_with_message_history = (
            RunnablePassthrough.assign(
                messages_summarized=self.summarize_ongo_messages, 
                context=self.parse_retriever_input | self.time_retriever).assign(
                    answer=self.document_chain_with_message_history)
        )

        # assign document chain to self
        self.retrieval_document_chain_with_message_history = retrieval_document_chain_with_message_history

    def run_chat(self, input: str) -> [Dict, Any]:

        # run chat with callback
        with get_openai_callback() as cb:
            result = self.retrieval_document_chain_with_message_history.invoke(
            {
                'date_today': get_today(),
                'user_profile': self.user_profile,
                "input":input
            },
            config={"configurable": {"session_id": self.session_id}}
            )


        # fill store history full with the input user and chatbot messages
        self.store_history[self.user_id][self.session_id]['full'].add_message(HumanMessage(input))
        self.store_history[self.user_id][self.session_id]['full'].add_message(AIMessage(result['answer']))

        return result, cb


class NewJournal():
    def __init__(self, llm: Any, user_id: str, session_id: str, store_history: Dict) -> None:
        self.llm = llm
        self.store_history = store_history
        self.user_id = user_id
        self.session_id = session_id
        self.new_journal = None

    def get_new_journal(self) -> None:
        # get full messages from sate session
        full_messages = self.store_history[self.user_id][self.session_id]['full'].messages

        # define journal summarizer prompt
        summarization_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "user",
                    """
                    You are summarizer assitant focusing on job and career actions and thoughts of the user.

                    Summarize the user's chat history based on career concerns to create a new personal journal observation about the user. 

                    Do NOT exceed 300 words.
                    """,
                ),
            ]
        )

        # instance the summarization chain
        summarization_chain = summarization_prompt | self.llm

        # invoke the chain
        new_journal = summarization_chain.invoke({"chat_history": full_messages})

        # add date of today to the new journal
        new_journal = f"date {get_today().strftime('%m/%d/%Y')} - {new_journal}"

        # assign document chain to self
        self.new_journal = new_journal


class UpdateStore():
    def __init__(self, journal_csv_path: str, chat_history_csv_path: str, new_journal: str, store_history: Dict, user_id: str, session_id: str) -> None:
        self.journal_csv_path = journal_csv_path
        self.chat_history_csv_path = chat_history_csv_path
        self.new_journal = new_journal
        self.store_history = store_history
        self.session_id = session_id
        self.user_id = user_id

    def update_store_journal_csv(self) -> None:
        # load pandas journal.csv
        df = pd.read_csv(self.journal_csv_path)
        # add new line at date of today with the summary message
        r = pd.DataFrame({str(len(df)+1):{'date': get_today(), 'sentence': self.new_journal}}).T
        r['session_id'] = self.session_id 
        df = pd.concat([df, r], ignore_index=True)
        # save the new dataframe
        df.to_csv(self.journal_csv_path, index=False)

    def update_store_chat_history(self) -> None:
        # new chat history
        new_df_chat_history = pd.DataFrame(self.store_history[self.user_id][self.session_id]['full'].dict()['messages'])
        new_df_chat_history['date'] = get_today()
        new_df_chat_history = new_df_chat_history[['date', 'content', 'type']]
        new_df_chat_history['sesion_id'] = self.session_id
        # old chat history
        df_chat_history = pd.read_csv(self.chat_history_csv_path, index_col=0)
        # add new_df_chat_messages to df_chat_messages
        df_chat_history = pd.concat([df_chat_history, new_df_chat_history], ignore_index=True).reset_index(drop=True)
        # save the new dataframe
        df_chat_history.to_csv(self.chat_history_csv_path, index=False)


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

