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
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, openai_api_key=self.api_key)
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
                You are a summarizer assistant focused on job and career actions and thoughts. 
                Your task is to summarize user information based on the provided context related to career concerns. 
                Ensure your summary does not exceed 300 words. 
                If the context provided is empty, return an empty string.

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
                    HumanMessage(content="Make a summary text of the user's answers to the formulaire. Starts with name and date of birth.")
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
                    HumanMessage(content="Make a summary text of your observations about the user.")
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
                    You are tasked with updating the user summary based on details from a journal summary. 
                    Begin with the user's name and date of birth, provided at the start of the journal summary. 
                    Your update should reflect the user's actions and thoughts related to their job and career development, as highlighted in the journal. 
                    Keep your summary concise and focused on professional growth, ensuring it does not exceed 300 words.

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
                As a career coach who speaks like a good old friend, you bring warmth and understanding to each interaction. 
                Stay concise and focus on both the job search and the user’s psychological well-being. 
                Proactively ask questions to explore not only career-related concerns but also emotional states and personal motivations. 
                Offer supportive feedback based on their input, career aspirations, and emotional needs, and gently guide them if they seem off-track.

                Refer to the following information in your responses:
                - Today's date: {date_today}
                - Updated user profile: {user_profile}
                - Context of the conversation: {context}

                Limit your responses to 50 tokens.

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
        if len(stored_messages)<=5:
            return False

        else:
            # summarization prompt
            summarization_prompt = ChatPromptTemplate.from_messages(
                [
                    MessagesPlaceholder(variable_name="chat_history"),
                    (
                        "user",
                        """

                        Distill the above chat messages into a single summary message. Include as many specific details as you can.

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

            # add last 5 messages to ongoing messages
            for message in self.store_history[self.user_id][self.session_id]['full'].messages[-5:]:
                self.store_history[self.user_id][self.session_id]['ongo'].add_message(message)

            # add summary message to ongoing messages
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