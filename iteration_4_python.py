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
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI

from langchain_pinecone import PineconeVectorStore

import faiss

import datetime

import pandas as pd

import re

import pprint

session_id = '0'

store_history = {
        session_id:
        {
            "ongo":ChatMessageHistory(), 
            "full":ChatMessageHistory()
        }
}


# function to set user and its session_id
def set_user(user, session_id):
    user = user
    session_id = session_id
    return user, session_id


# function to get the date of today 
def get_today():
    return datetime.datetime.now()


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
def load_csv_from_directory(directory, csv_file_name):
    # exclusif function to add the date to the csv documents loaded
    def add_date_to_documents(docs):
        for i, doc in enumerate(docs):
            page_content = doc.page_content
            try:
                date = re.search(r'\d{2}/\d{2}/\d{4}', page_content).group() 
                # date = datetime.datetime.strptime(date, '%m/%d/%Y')
                doc.metadata["created_at"] = get_today()
            except:
                continue
        return docs


    loader = DirectoryLoader(directory, glob=csv_file_name, loader_cls=CSVLoader)
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


def get_user_profile(llm, user_summary, journal_summary):
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
        user_profile = get_user_profile(llm, user_summary, journal_summary)
        return user_profile


# functions for time weighted retriever
def get_faiss_time_retriever(embedding, journal_docs):
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

    for i, doc in enumerate(journal_docs):
        page_content = doc.page_content
        metadata = doc.metadata
        faiss_time_retriever.add_documents([Document(page_content=page_content, metadata=metadata)])

    return pinecone_time_retriever


# function for chatbot
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


def get_document_chain_with_message_history(document_chain):

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


def get_retrieval_document_chain_with_message_history(document_chain_with_message_history, time_retriever, llm):
    def summarize_messages(chain_input):
        stored_messages = store_history[session_id]['ongo'].messages
        print('ongo-------------------')
        pprint.pprint(store_history[session_id]['ongo'].dict())
        print('-------------------')
        if len(stored_messages) <=2:
            store_history[session_id]['full'].add_message(stored_messages[-1])
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
        store_history[session_id]['ongo'].add_message(stored_messages[-2])
        store_history[session_id]['ongo'].add_message(stored_messages[-1])
        store_history[session_id]['full'].add_message(stored_messages[-2])
        store_history[session_id]['full'].add_message(stored_messages[-1])

        print('full-------------------')
        pprint.pprint(store_history[session_id]['full'].dict())
        print('-------------------')

        return True

    from typing import Dict
    def parse_retriever_input(params: Dict):
        return params["input"]

    retrieval_document_chain_with_message_history = (
        RunnablePassthrough.assign(
            messages_summarized=summarize_messages, 
            context=parse_retriever_input | time_retriever).assign(
                answer=document_chain_with_message_history)
    )

    return retrieval_document_chain_with_message_history


def run_chat(input, user_profile, retrieval_document_chain_with_message_history):
    with get_openai_callback() as cb:
        result = retrieval_document_chain_with_message_history.invoke(
        {
            'date_today': get_today(),
            'user_profile': user_profile,
            "input":input
        },
        config={"configurable": {"session_id": session_id}}
        )
    return result, cb


# function to make new journal row
def get_new_journal(llm):
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
def update_store_journal_csv(journal_csv_path, new_journal):
    # load pandas journal.csv
    df = pd.read_csv(journal_csv_path)
    # add new line at date of today with the summary message
    r = pd.DataFrame({str(len(df)+1):{'date': get_today(), 'sentence': new_journal.content}}).T
    df = pd.concat([df, r], ignore_index=True)
    # save the new dataframe
    df.to_csv(journal_csv_path, index=False)


def update_faiss_time_retriever(journal_csv_path, new_journal, faiss_time_retriever):
    # add in the pinecone time weighted retriever
    df = pd.read_csv(journal_csv_path)
    metadata = {
        'source': journal_csv_path,
        'row': len(df),
        "created_at": get_today()
    }
    faiss_time_retriever.add_documents([Document(page_content=new_journal.content, metadata=metadata)])


def update_store_chat_history(chat_history_csv_path):
    # new chat history
    new_df_chat_history = pd.DataFrame(store_history[session_id]['full'].dict()['messages'])
    new_df_chat_history['date'] = get_today()
    new_df_chat_history = new_df_chat_history[['date', 'content', 'type']]
    # old chat history
    df_chat_history = pd.read_csv(chat_history_csv_path, index_col=0)
    # add new_df_chat_messages to df_chat_messages
    df_chat_history = pd.concat([df_chat_history, new_df_chat_history], ignore_index=True).reset_index(drop=True)
    # save the new dataframe
    df_chat_history.to_csv(chat_history_csv_path, index=False)



import streamlit as st
import datetime
import pandas as pd
from pathlib import Path
import tempfile
import os

# initialize session state with all the variables we need

temp_dir = None
llm = None
embedding = None
journal_csv_path = None
formulaire_csv_path = None
chat_history_csv_path = None
formulaire_docs = None
journal_docs = None
history_docs = None
user_profile = None
faiss_time_retriever = None
retrieval_document_chain_with_message_history = None
new_journal = None

with st.sidebar:
    llm_name_model = st.selectbox('Pick a llm model',('openai', 'anthropic', 'mistralai'))
    embedding_name_model = st.selectbox('Pick a embedding model',('openai','none'))        
    llm_api_key = st.text_input("LLM API Key", key="chatbot_api_key", type="password")

    if llm_name_model and llm_api_key and embedding_name_model: 
        # select models
        llm = select_llm_model(llm_name_model)
        embedding = select_embedding_model(embedding_name_model)
        
        # feed st session state
        st.session_state['llm'] = llm
        st.session_state['embedding'] = embedding
    else:
        st.error("Please enter the LLM API key.")

    # upload files
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


            # Initialize docs from uploaded files
            if 'journal.csv' in uploaded_file.name:
                journal_csv_path = os.path.join(temp_dir, uploaded_file.name)
                

            elif 'formulaire.csv' in uploaded_file.name:
                formulaire_csv_path = os.path.join(temp_dir, uploaded_file.name)
                

            elif 'history.csv' in uploaded_file.name:
                chat_history_csv_path = os.path.join(temp_dir, uploaded_file.name)
                


    # feed st session state
    st.session_state['temp_dir'] = temp_dir
    st.session_state['journal_csv_path'] = journal_csv_path
    st.session_state['formulaire_csv_path'] = formulaire_csv_path
    st.session_state['chat_history_csv_path'] = chat_history_csv_path



# Main chat interface
st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")

if st.button('Start Chat'):
    # load models
    llm = st.session_state['llm']
    embedding = st.session_state['embedding']
    temp_dir = st.session_state['temp_dir']

    # load documents to improve
    formulaire_docs = load_csv_from_directory(temp_dir, 'formulaire.csv')
    journal_docs = load_csv_from_directory(temp_dir, 'journal.csv')
    history_docs = load_csv_from_directory(temp_dir, 'history.csv')
    st.success("CSV files loaded successfully.")

    # generate user profile
    user_profile = user_profile_generation(llm, formulaire_docs, journal_docs)
    st.success("User profile generated successfully.")
    st.write(f"User Profile: {user_profile}")

    # create time weighted vector store retriever
    faiss_time_retriever = get_faiss_time_retriever(embedding, journal_docs)
    st.success("Time weighted vector store retriever created successfully.")

    # create rag chain chatbot
    document_chain = get_document_chain(llm)
    document_chain_with_message_history = get_document_chain_with_message_history(document_chain)
    retrieval_document_chain_with_message_history = get_retrieval_document_chain_with_message_history(document_chain_with_message_history, faiss_time_retriever, llm)
    st.success("Rag chain chatbot created successfully.")

    # start chat
    st.write("Chat started.")

    # feed st session state
    st.session_state['user_profile'] = user_profile
    st.session_state['faiss_time_retriever'] = faiss_time_retriever
    st.session_state['retrieval_document_chain_with_message_history'] = retrieval_document_chain_with_message_history


# End chat and handle file updates
if st.button('End Chat'):
    st.session_state['chat_active'] = False
    # load session state
    llm = st.session_state['llm']
    journal_csv_path = st.session_state['journal_csv_path']
    chat_history_csv_path = st.session_state['chat_history_csv_path']
    faiss_time_retriever = st.session_state['faiss_time_retriever'] 

    # update journal with full history
    new_journal = get_new_journal(llm)
    st.write(f"New Journal Update: {new_journal.content}")

    # update journal in csv
    update_store_journal_csv(journal_csv_path, new_journal)
    st.success("Journal updated successfully.")

    # update journal in faiss time retriever
    update_faiss_time_retriever(journal_csv_path, new_journal, faiss_time_retriever)
    st.success("Faiss time retriever updated successfully.")

    # update chat history with full history
    update_store_chat_history(chat_history_csv_path)
    st.success("Chat history updated successfully.")

    # download updated journal and history csv
    st.download_button(
        label="Download Updated Journal CSV",
        data=pd.read_csv(journal_csv_path),
        file_name="updated_journal.csv",
        mime="text/csv"
    )

    st.download_button(
        label="Download Updated History CSV",
        data=pd.read_csv(chat_history_csv_path),
        file_name="updated_history.csv",
        mime="text/csv"
    )

    # feed st session state
    st.session_state['new_journal'] = new_journal
    st.session_state['faiss_time_retriever'] = faiss_time_retriever


# activate chat
st.session_state['chat_active'] = True
st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
retrieval_document_chain_with_message_history = st.session_state['retrieval_document_chain_with_message_history']
user_profile = st.session_state['user_profile']
    
# Chat interaction
if st.session_state.get('chat_active', False):
    for msg in st.session_state['messages']:
        st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input("Your message"):

    # load session state
    
    if not retrieval_document_chain_with_message_history:
        st.error("Please make sure all configurations are properly set.")
    else:
        st.session_state['messages'].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response, cb = run_chat(prompt, user_profile, retrieval_document_chain_with_message_history)
        answer = response['answer']
        print('-------')
        
        pprint.pprint(store_history)
        print('-------')
        st.session_state['messages'].append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)


