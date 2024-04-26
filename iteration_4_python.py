from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.prompts import MessagesPlaceholder, PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate

from langchain.schema.runnable import RunnablePassthrough

from langchain.retrievers import TimeWeightedVectorStoreRetriever
from PineconeModifiedTimeWeightedRetriever import Pinecone_Modified_TimeWeightedVectorStoreRetriever

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_community.callbacks import get_openai_callback

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.document_loaders import BaseLoader
from langchain_core.runnables.history import RunnableWithMessageHistory , RunnableBranch
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic, AnthropicEmbeddings
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from langchain_pinecone import PineconeVectorStore

import faiss

import datetime

import pandas as pd

import re


# function to set user and its session_id
def set_user(user, session_id):
    user = user
    session_id = session_id
    return user, session_id


# function to get the date of today 
def get_today():
    return datetime.datetime.now().strftime('%m/%d/%Y')


# function to select the llm chat model
def select_llm_model(llm_model_name):
    if llm_model_name == "openai":
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    if llm_model_name == "anthropic":
        llm = ChatAnthropic(model='claude-3-opus-20240229')
    if llm_model_name == "mistralai":
        llm = ChatMistralAI(model='open-mistral-7b')
    return llm


def select_embedding_model(embedding_model_name):
    if embedding_model_name == "openai":
        embedding = OpenAIEmbeddings()
    if embedding_model_name == "anthropic":
        embedding = AnthropicEmbeddings()
    if embedding_model_name == "mistralai":
        embedding = MistralAIEmbeddings()
    return embedding


# load documents from a directory
def load_csv_from_directory(directory):
    # exclusif function to add the date to the csv documents loaded
    def add_date_to_documents(docs):
        for i, doc in enumerate(docs):
            page_content = doc.page_content
            try:
                date = re.search(r'\d{2}/\d{2}/\d{4}', page_content).group() 
                date = datetime.datetime.strptime(date, '%m/%d/%Y')
                doc.metadata["created_at"] = date
            except:
                continue
        return docs


    loader = DirectoryLoader(directory, glob="*.csv", loader_cls=CSVLoader)
    docs = loader.load()
    docs = add_date_to_documents(docs)    
    return docs


# functions for updated user profile generation
def make_summarizer(llm):
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
    summarizer = create_stuff_documents_chain(llm, summary_prompt)
    return summarizer


def get_user_summary(summarizer, formulaire_docs):
    user_summary = summarizer.invoke(
        {
            "context": formulaire_docs,
            "messages": [
                HumanMessage(content="Make a summary text of the user's answers to the formulaire. Starts with name and date of birth. do not exeed 300 words.")
            ],
        }
    )
    return user_summary


def get_journal_summary(summarizer, journal_docs):
    journal_summary = summarizer.invoke(
        {
            "context": journal_docs,
            "messages": [
                HumanMessage(content="Make a summary text of your observations about the user. do not exeed 300 words.")
            ],
        }
    )
    return journal_summary


    user_profiler_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_user_profile_prompt_instructions + \
                """
                User Summary:
                {user_summary}

                Journal Summary:
                {journal_summary}

                """,
            ),
        ]
    )
    user_profiler_chain = user_profiler_prompt | llm
    return user_profiler_chain


def get_user_profile(user_summary, journal_summary):
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
    user_profiler_chain = user_profiler_prompt | llm


    user_profile = user_profiler_chain.invoke(
        {
            "user_summary": user_summary, 
            "journal_summary": journal_summary
        }
        ).content
    return user_profile


def user_profile_generation(llm, formulaire_docs, journal_docs):
    summarizer = make_summarizer(llm)
    user_summary = get_user_summary(summarizer, formulaire_docs)
    if len(journal_docs) == 0:
        return user_summary
    else:
        journal_summary = get_journal_summary(summarizer, journal_docs)
        user_profile = get_user_profile(user_summary, journal_summary)
        return user_profile


# functions for time weighted retriever
def faiss_time_retriever(embedding, journal_docs):
    faiss_index = faiss.IndexFlatL2(1536)
    faiss_vectorstore = FAISS(embedding, faiss_index, InMemoryDocstore({}), {})
    faiss_time_retriever = TimeWeightedVectorStoreRetriever(
        vectorstore=faiss_vectorstore, 
        decay_rate=1e-5, 
        k=5
    )

    # no rest api so fill data evry time creation
    for i, doc in enumerate(journal_docs):
        page_content = doc.page_content
        metadata = doc.metadata
        faiss_time_retriever.add_documents([Document(page_content=page_content, metadata=metadata)])

    return faiss_time_retriever


def pinecone_time_retriever(embedding, index_name):
    pinecone_vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding)
    pinecone_time_retriever = Pinecone_Modified_TimeWeightedVectorStoreRetriever(
        vectorstore=pinecone_vectorstore, 
        decay_rate=1e-5, 
        k=5
    )

    return pinecone_time_retriever


# function for chatbot
store_history = {
        session_id:
        {
            "ongo":ChatMessageHistory(), 
            "full":ChatMessageHistory()
        }
    }


def get_document_chain(llm):
    # coach chatbot prompt
    coach_chatbot_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
            """
            You are  coach for job and career search.

            The user speaks to and you speak to him.

            The conversation is like normale frienship conversation.

            Stay concise and to the point.

            If the user is asking to describe or explain more you can say more but keep short.

            Your wisdom should guide the user clearly and confidently, lighting the way to a fulfilling career journey.

            However, you are capable of jugment on user input related to career search and his profile.

            If you think the user is not in the right direction, you can tell him.

            The provided chat history summary includes facts about the user you are speaking with.

            this is the date of today conversation: 
            {date_today}

            this is the updated user profile to refer to: 
            {user_profile}

            this is the context to refer to:
            {context}

            """
            ),

            MessagesPlaceholder("chat_history"),

            ("human", "{input}"),
        ]
        )

    # define document chain
    document_chain = create_stuff_documents_chain(llm, coach_chatbot_prompt)

    return document_chain


def get_document_chain_with_message_history(document_chain, store_history):

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store_history:
            store_history[session_id] = {
            "ongo":ChatMessageHistory(), 
            "full":ChatMessageHistory()
            }
        return store_history[session_id]['ongo']


    document_chain_with_message_history = RunnableWithMessageHistory(
        document_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        answer_messages_key="answer"
    )

    return document_chain_with_message_history


def get_retrieval_document_chain_with_message_history(document_chain_with_message_history, pinecone_time_retriever, llm, store_history):
    def summarize_messages(chain_input):
        stored_messages = store_history[session_id]['ongo'].messages
        if len(stored_messages) == 0:
            return False
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
                ),
            ]
         )

        summarization_chain = summarization_prompt | llm
        summary_message = summarization_chain.invoke({"chat_history": stored_messages})

        # each chat summarization - manage history storing
        store_history[session_id]['ongo'].clear()
        store_history[session_id]['ongo'].add_message(summary_message)
        store_history[session_id]['full'].add_message(stored_messages[-2])
        store_history[session_id]['full'].add_message(stored_messages[-1])

        return True

    from typing import Dict
    def parse_retriever_input(params: Dict):
        return params["input"]

    retrieval_document_chain_with_message_history = (
        RunnablePassthrough.assign(
            messages_summarized=summarize_messages, 
            context=parse_retriever_input | pinecone_time_retriever).assign(
                answer=document_chain_with_message_history)
    )

    return retrieval_document_chain_with_message_history


def run_chat(session_id, input, retrieval_document_chain_with_message_history):
    with get_openai_callback() as cb:
        result = retrieval_document_chain_with_message_history.invoke(
        {
            'date_today': date_today,
            'user_profile': user_profile,
            "input":input
        },
        config={"configurable": {"session_id": session_id}}
        )
    return result, cb


# function to make new journal row
def get_new_journal(llm, store_history):
    # Summarize chat history
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
    summarization_chain = summarization_prompt | llm
    new_journal = summarization_chain.invoke({"chat_history": store_history[session_id]['full'].messages})

    return new_journal


# function to update and store
def update_store_journal(journal_csv_path, pinecone_time_retriever, new_journal):
    # load pandas journal.csv
    df = pd.read_csv(journal_csv_path)
    # add new line at date of today with the summary message
    r = pd.DataFrame({str(len(df)+1):{'date': date_today, 'sentence': new_journal.content}}).T
    df = pd.concat([df, r], ignore_index=True)
    # save the new dataframe
    df.to_csv(journal_csv_path, index=False)


    # add in the pinecone time weighted retriever
    metadata = {
        'source': journal_csv_path,
        'row': len(df),
        "created_at": datetime.datetime.now()
    }
    pinecone_time_retriever.add_documents([Document(page_content=new_journal.content, metadata=metadata)])


def update_store_chat_history(chat_history_csv_path, store_history)
    # new chat history
    new_df_chat_history = pd.DataFrame(store_history[session_id]['full'].dict()['messages'])
    new_df_chat_history['date'] = date_today
    new_df_chat_history = new_df_chat_history[['date', 'content', 'type']]
    # old chat history
    df_chat_history = pd.read_csv(chat_history_csv_path, index_col=0)
    # add new_df_chat_messages to df_chat_messages
    df_chat_history = pd.concat([df_chat_history, new_df_chat_history], ignore_index=True).reset_index(drop=True)
    # save the new dataframe
    df_chat_history.to_csv(chat_history_csv_path, index=False)


