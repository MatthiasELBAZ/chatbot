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
from dotenv import load_dotenv
load_dotenv()

# utility functions
def get_today():
    return datetime.datetime.now()


class UserSessionIdentity():
    def __init__(self, user_id: str, session_id: str, user_starting_date: str) -> None:
        self._user_id = user_id
        self._session_id = session_id
        self._user_starting_date = user_starting_date

    @property
    def user_id(self) -> str:
        """Return the user ID."""
        return self._user_id

    @property
    def session_id(self) -> str:
        """Return the session ID."""
        return self._session_id

    @property
    def user_starting_date(self) -> str:
        """Return the user starting date."""
        return self._user_starting_date


class UserSessionStoreHistory(UserSessionIdentity):
    def __init__(self, user_id: str, session_id: str, user_starting_date: str) -> None:
        super().__init__(user_id, session_id, user_starting_date)
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

    def get_ongo_history(self, user_id, session_id) -> ChatMessageHistory:
        # Retrieve the chat message history for the ongoing session, initializing if necessary.
        return self.store_history[user_id][session_id]['ongo']

    def get_full_history(self, user_id, session_id) -> ChatMessageHistory:
        # Retrieve the chat message history for the full session, initializing if necessary.
        return self.store_history[user_id][session_id]['full']


class UserDirectoryLoader():
    def __init__(self, directory: str) -> None:
        self.directory = directory
    
    def add_date_to_documents(self, docs: list, user_starting_date: str) -> list:
        for doc in docs:
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

    def load_csv_from_directory(self, csv_file_name: str, user_starting_date: str) -> list:
        loader = DirectoryLoader(self.directory, glob=csv_file_name, loader_cls=CSVLoader)
        docs = loader.load()
        docs = self.add_date_to_documents(docs, user_starting_date)    
        return docs


class LoadModels():
    def __init__(self, llm_api_key: str, embedding_api_key: str, llm_name_model: str, embedding_name_model: str, temperature: float, max_tokens: int) -> None:
        self.llm_name_model = llm_name_model
        self.embedding_name_model = embedding_name_model
        self.llm_api_key = llm_api_key
        self.embedding_api_key = embedding_api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    def select_llm_model(self) -> Any:
        if self.llm_name_model == "openai_3.5":
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=self.temperature, max_tokens=self.max_tokens, openai_api_key=self.llm_api_key)
        if self.llm_name_model == "openai_4":
            llm = ChatOpenAI(model="gpt-4-turbo", temperature=self.temperature, max_tokens=self.max_tokens, openai_api_key=self.llm_api_key)
        if self.llm_name_model == "anthropic_opus":
            llm = ChatAnthropic(model='claude-3-opus-20240229', temperature=self.temperature, max_tokens=self.max_tokens, anthropic_api_key=self.llm_api_key)
        if self.llm_name_model == "anthropic_sonnet":
            llm = ChatAnthropic(model='claude-3-sonnet-20240229', temperature=self.temperature, max_tokens=self.max_tokens, anthropic_api_key=self.llm_api_key)
        if self.llm_name_model == "mistral_large":
            llm = ChatMistralAI(model='mistral-large-latest', temperature=self.temperature, max_tokens=self.max_tokens, mistral_api_key=self.llm_api_key)
        if self.llm_name_model == "mistral_8x22B":
            llm = ChatMistralAI(model='open-mixtral-8x22b', temperature=self.temperature, max_tokens=self.max_tokens, mistral_api_key=self.llm_api_key)
        return llm

    def select_embedding_model(self) -> Any:
        if self.embedding_name_model == "openai_3.5" or self.embedding_name_model == "openai_4":
            embedding = OpenAIEmbeddings(openai_api_key=self.embedding_api_key)
        if self.embedding_name_model == "mistral_large" or self.embedding_name_model == "mistral_8x22B":
            embedding = MistralAIEmbeddings(api_key=self.embedding_api_key)
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
                <instructions>
                You are a summarizer assistant focused on job and career actions and thoughts. 
                Your task is to make a user profile description based on the provided context. 
                Ensure your summary does not exceed 300 words. 
                If the context provided is empty, return an empty string.
                </instructions>

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
                    HumanMessage(content="Make a summary text of the user's answers to the formulaire. Starts with name and date of birth and personal information then make a psychologic portrait of the user.")
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
                    HumanMessage(content="Make a summary text of your observations about the user by emphazing the journey of the user.")
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
                    <instructions>
                    You are tasked with updating the user summary based on details from a journal summary. 
                    Begin with the user's name and date of birth, provided at the start of the journal summary. 
                    Your update should reflect the user's actions and thoughts related to their job and career development, as highlighted in the journal. 
                    Keep your summary concise and focused on professional growth, ensuring it does not exceed 300 words.
                    </instructions>
                    
                    <user_summary>
                    {user_summary}
                    </user_summary>

                    <journal_summary>
                    {journal_summary}
                    </journal_summary>

                    """
                ),
                # add for anthropic
                MessagesPlaceholder(variable_name="messages"),
                
            ]
        )

        # define user profiler chain
        user_profiler_chain = user_profiler_prompt | self.llm

        # invoke user profiler
        user_profile = user_profiler_chain.invoke(
            {
                "user_summary": user_summary, 
                "journal_summary": journal_summary,
                # add for anthropic
                "messages": [
                    HumanMessage(content="Make the updated user's profile based on the user summary and journal summary.")
                ],
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
    def __init__(self, embedding: Any, type: str, decay_rate: float, k: int) -> None:
        self.embedding = embedding
        self.type = type
        self.vectorstore = None
        self.time_retriever = None
        self.decay_rate = decay_rate
        self.k = k

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
                decay_rate=self.decay_rate, 
                k=self.k
            )
        if self.type == "pinecone":
            time_retriever = Pinecone_Modified_TimeWeightedVectorStoreRetriever(
                vectorstore=self.vectorstore, 
                decay_rate=self.decay_rate, 
                k=self.k
            )

        # assign time retriever to self
        self.time_retriever = time_retriever

    def time_retriever_add_documents(self, docs: list) -> None:
        for i, doc in enumerate(docs):
            page_content = doc.page_content
            metadata = doc.metadata
            self.time_retriever.add_documents([Document(page_content=page_content, metadata=metadata)])


class AnthropicTokenCounter(BaseCallbackHandler):
    def __init__(self, llm):
        self.llm = llm
        # get model name
        try:
            self.llm_name_model = llm.model
        except:
            self.llm_name_model = llm.model_name
        
        # get price per token
        if 'gpt-3.5' in self.llm_name_model:
            self.price_input = 0.5/1e6
            self.price_output = 1.5/1e6
        elif 'gpt-4' in self.llm_name_model:
            self.price_input = 10/1e6
            self.price_output = 30/1e6
        elif 'sonnet' in self.llm_name_model:
            self.price_input = 3/1e6
            self.price_output = 15/1e6
        elif 'opus' in self.llm_name_model:
            self.price_input = 15/1e6
            self.price_output = 75/1e6
        elif 'large' in self.llm_name_model:
            self.price_input = 4/1e6
            self.price_output = 12/1e6
        elif '8x22' in self.llm_name_model:
            self.price_input = 2/1e6
            self.price_output = 6/1e6
        else:
            self.price_input = 0
            self.price_output = 0
        
        
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


class RetrievalDocumentChainMemory(UserSessionStoreHistory):
    def __init__(self, llm: Any, time_retriever: Any, user_profile: str, user_id: str, session_id: str, store_history: Dict, user_starting_date: str, buffer_num_ongo_messages: int) -> None:
        super().__init__(user_id, session_id, user_starting_date)
        self.time_retriever = time_retriever
        self.user_profile = user_profile
        self.store_history = store_history
        self.document_chain = None
        self.document_chain_with_message_history = None
        self.retrieval_document_chain_with_message_history = None
        self.llm = llm
        self.buffer_num_ongo_messages = buffer_num_ongo_messages

    def parse_retriever_input(self, params: Dict) -> str:
        # return input from user
        return params["input"]
    
    def get_document_chain(self) -> None:
        # define chatbot prompt
        coach_chatbot_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                """
                <instructions>
                You are a friend of the user and expert in career coaching. You speak with a cool tone making short sentences. Stay focus on both the job search and the user’s psychological well-being. Proactively ask questions to explore his concerns, emotional states and personal motivations. Offer feedback based on its input, career aspirations, and emotional needs, and guide them if he seems off-track. You do not speak too much and if details are needed, you prioritize them and talk about the most important first. Refer to the following information in your responses:
                - Today's date
                - Updated user profile
                - RAG Context
                - Previous chat history
                </instructions>

                <example>
                -user: I'm feeling overwhelmed by my job search.
                -bot: It's normal to feel overwhelmed. Let's break it down. What's the most challenging part for you right now?
                </example>

                <example>
                -user: I'm tired of looking for a job.
                -bot: "what are you saying here my friend? Wake you up and let's go to work! Shine for me!!
                </example>

                <example>
                -user: Can you detailed this.
                -bot: Yes sure let s start with the most important point then move to the other details if you want.
                </example>
                """
                ),

                MessagesPlaceholder("chat_history"),

                ("human", 
                
                """
                <instructions>
                With the following information, you anwer to my questions as user input as friend and career coach.
                </instructions>

                <date_today>
                {date_today}
                </date_today>

                <user_profile>
                {user_profile}
                </user_profile>

                <rag_context>
                {context}
                </rag_context>

                <user_input>
                {input}
                </user_input>

                """),
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
        if len(stored_messages)<=self.buffer_num_ongo_messages:
            return False

        else:
            # summarization prompt
            summarization_prompt = ChatPromptTemplate.from_messages(
                [
                    MessagesPlaceholder(variable_name="chat_history"),
                    (
                        "user",
                        """
                        <instructions>
                        Distill the above chat messages into a single summary message. Include as many specific details as you can. Mkae short sentence for using maximum 300 words.
                        </instructions>
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

            # add last buffer_num_ongo_messages messages to ongoing messages 
            for message in self.store_history[self.user_id][self.session_id]['full'].messages[-self.buffer_num_ongo_messages:]:
                self.store_history[self.user_id][self.session_id]['ongo'].add_message(message)

            # add human message before summary message for anthropic model
            self.store_history[self.user_id][self.session_id]['ongo'].add_message(HumanMessage("Make a summary of our chat messages unitl now."))

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
        # get callbacker
        callbacker = AnthropicTokenCounter(self.llm)

        # run chat with callback
        result = self.retrieval_document_chain_with_message_history.invoke(
            {
                'date_today': get_today(),
                'user_profile': self.user_profile,
                "input":input
            },
            config={
                "configurable": {"session_id": self.session_id}, 
                "callbacks": [callbacker]
            }
        )


        # fill store history full with the input user and chatbot messages
        self.store_history[self.user_id][self.session_id]['full'].add_message(HumanMessage(input))
        self.store_history[self.user_id][self.session_id]['full'].add_message(AIMessage(result['answer']))

        return result, {'input_tokens':callbacker.input_tokens, 'output_tokens':callbacker.output_tokens, 'price':callbacker.get_request_price()}


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
                    <instructions>
                    You are summarizer assitant focusing on job and career actions and thoughts of the user.

                    Summarize the user's chat history based on career concerns to create a new personal journal observation about the user. 

                    Do NOT exceed 300 words.
                    </instructions>
                    """,
                ),
            ]
        )

        # instance the summarization chain
        summarization_chain = summarization_prompt | self.llm

        # invoke the chain
        new_journal = summarization_chain.invoke({"chat_history": full_messages})

        # assign document chain to self
        self.new_journal = new_journal.content


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