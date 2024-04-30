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
import streamlit as st
import datetime
from pathlib import Path
import tempfile
import os


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

    # get session user history from app state
    session_id = st.session_state.session_id
    store_history = st.session_state.store_history

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
    
    # get session user history from app state
    session_id = st.session_state.session_id
    store_history = st.session_state.store_history

    def summarize_messages(chain_input):

        # get last messages
        stored_messages = store_history[session_id]['ongo'].messages

        # do nothing if no messages
        if len(stored_messages)==0:
            return False

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
        summarization_chain = summarization_prompt | llm
        summary_message = summarization_chain.invoke({"chat_history": stored_messages})
        
        # clear ongoing messages and add summary message
        store_history[session_id]['ongo'].clear()
        store_history[session_id]['ongo'].add_message(summary_message)

        # modify store history in app state
        st.session_state['store_history'] = store_history
        
        return True

    # create the full chatbot chain
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

    # get session id from state session
    session_id = st.session_state.session_id
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

    # get full messages from sate session
    full_messages = st.session_state.store_history[st.session_state.session_id]['full'].messages

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
    new_journal = summarization_chain.invoke({"chat_history": full_messages})

    return new_journal.content

# function to update and store
def update_store_journal_csv(journal_csv_path, new_journal):
    # load pandas journal.csv
    df = pd.read_csv(journal_csv_path)
    # add new line at date of today with the summary message
    r = pd.DataFrame({str(len(df)+1):{'date': get_today(), 'sentence': new_journal}}).T
    df = pd.concat([df, r], ignore_index=True)
    # save the new dataframe
    df.to_csv(journal_csv_path, index=False)

def update_store_chat_history(chat_history_csv_path, store_history, session_id):
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


# function to initialize session state
def initialize_session_state():
    """Initialize all the required session state variables."""
    keys = [
        "llm", "embedding", "temp_dir", "journal_csv_path", 
        "formulaire_csv_path", "chat_history_csv_path", "user_profile",
        "faiss_time_retriever", "retrieval_document_chain_with_message_history",
        "journal_docs", "formulaire_docs", "history_docs", 'new_journal'
    ]
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = None

def handle_file_upload(uploaded_files):
    """Handle the uploading and saving of files."""
    if uploaded_files:
        # st.write("Uploaded Files:")
        for uploaded_file in uploaded_files:
            file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            # st.write(f"{uploaded_file.name} saved at {file_path}")
            categorize_file(uploaded_file.name, file_path)

def categorize_file(file_name, file_path):
    """Categorize and save paths for uploaded files based on their type."""
    if 'journal.csv' in file_name:
        st.session_state.journal_csv_path = file_path
    elif 'formulaire.csv' in file_name:
        st.session_state.formulaire_csv_path = file_path
    elif 'history.csv' in file_name:
        st.session_state.chat_history_csv_path = file_path

def setup_sidebar():
    """Set up sidebar elements for model selection and file uploading."""
    session_id = st.sidebar.text_input("Session ID", key="session_id")
    llm_name_model = st.sidebar.selectbox('Pick a LLM model', ('openai', 'anthropic', 'mistralai'))
    embedding_name_model = st.sidebar.selectbox('Pick an embedding model', ('openai', 'none'))        
    llm_api_key = st.sidebar.text_input("LLM API Key", key="chatbot_api_key", type="password")

    if llm_name_model and llm_api_key and embedding_name_model and session_id: 
        st.session_state.llm = select_llm_model(llm_name_model)
        st.session_state.embedding = select_embedding_model(embedding_name_model)
    else:
        st.sidebar.error("Please enter the LLM API key.")
    
    uploaded_files = st.sidebar.file_uploader(
        "Choose a User folder and upload files", 
        accept_multiple_files=True, 
        type=['txt', 'csv', 'png', 'jpg', 'pdf']
    )
    if not st.session_state.temp_dir:
        st.session_state.temp_dir = tempfile.mkdtemp()

    with st.sidebar:
        handle_file_upload(uploaded_files)

# Function to handle the starting of the chat
def start_chat():
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

    # feed st session state
    st.session_state['user_profile'] = user_profile
    st.session_state['faiss_time_retriever'] = faiss_time_retriever
    st.session_state['retrieval_document_chain_with_message_history'] = retrieval_document_chain_with_message_history
    st.session_state['journal_docs'] = journal_docs
    st.session_state['formulaire_docs'] = formulaire_docs
    st.session_state['history_docs'] = history_docs

    # active chat
    st.session_state['chat_active'] = True
    st.write("Chat started.")
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Function to handle the ending of the chat
def end_chat():
    """Handle the end of a chat session."""
    # load session state
    llm = st.session_state['llm']
    journal_csv_path = st.session_state['journal_csv_path']
    chat_history_csv_path = st.session_state['chat_history_csv_path']
    store_history = st.session_state['store_history']
    session_id = st.session_state['session_id']

    # update journal with full history
    new_journal = get_new_journal(llm)
    if 'new_journal' not in st.session_state:
        st.session_state.new_journal = new_journal
    st.write(f"New Journal Update: {new_journal}")

    # update journal in csv
    update_store_journal_csv(journal_csv_path, new_journal)
    st.success("Journal updated successfully.")
    if 'updated_journal_true' not in st.session_state:
        st.session_state.updated_journal_true = True

    # update chat history with full history
    update_store_chat_history(chat_history_csv_path, store_history, session_id)
    st.success("Chat history updated successfully.")
    if 'updated_chat_history_true' not in st.session_state:
        st.session_state.updated_chat_history_true = True

    # deactivate chat
    st.session_state['chat_active'] = False
    st.write("Chat ended.")

# Main function   
def main():

    # initialize session state
    initialize_session_state()
    
    # setup sidebar
    setup_sidebar()

    # start chat session store history
    if 'store_history' not in st.session_state:
        if st.session_state.get('session_id', False):
            session_id = st.session_state.session_id

            st.session_state.store_history ={
                session_id: {
                "ongo": ChatMessageHistory(),
                "full": ChatMessageHistory()}
            
            }

    # start chat buttons
    if st.button('Start Chat'):
        start_chat()

    # end chat button
    if st.button('End Chat'):
        end_chat()

    # download updated journal and history csv
    if 'updated_journal_true' in st.session_state:
        journal_csv_path = st.session_state.journal_csv_path 
        st.download_button(
            label="Download Updated Journal CSV",
            data=pd.read_csv(journal_csv_path).to_csv(index=False),
            file_name="updated_journal.csv",
            mime="text/csv"
        )

    if 'updated_chat_history_true' in st.session_state:
        chat_history_csv_path = st.session_state.chat_history_csv_path
        st.download_button(
            label="Download Updated History CSV",
            data=pd.read_csv(chat_history_csv_path).to_csv(index=False),
            file_name="updated_history.csv",
            mime="text/csv"
        )

    # Chat interaction
    retrieval_document_chain_with_message_history = st.session_state['retrieval_document_chain_with_message_history']
    user_profile = st.session_state['user_profile']

    if "messages" in st.session_state:
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Your message"):
    
        if not retrieval_document_chain_with_message_history:
            st.error("Please make sure all configurations are properly set.")
        else:
            st.session_state['messages'].append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            response, cb = run_chat(prompt, user_profile, retrieval_document_chain_with_message_history)
            answer = response['answer']

            st.session_state['store_history'][st.session_state.session_id]['full'].add_message(HumanMessage(prompt))
            st.session_state['store_history'][st.session_state.session_id]['full'].add_message(AIMessage(answer))

            print('-----------------------------------')
            print('session_id', st.session_state.session_id)
            print('-----------------------------------')
            print('ongo dict')
            ongo_messages = st.session_state.store_history[st.session_state.session_id]['ongo'].messages
            pprint.pprint(ongo_messages)
            print(len(ongo_messages))
            print('-----------------------------------')
            print('full dict')
            full_messages = st.session_state.store_history[st.session_state.session_id]['full'].dict()
            pprint.pprint(full_messages)
            print(len(full_messages))
            print('-----------------------------------')
            
            st.session_state['messages'].append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
    

if __name__ == "__main__":
    main()

