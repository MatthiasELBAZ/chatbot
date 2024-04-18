

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import MessagesPlaceholder, PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.document_loaders import BaseLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.vectorstores import Chroma
JOB_CSV_PATH = "georgette/journal/job_search_journal.csv"
FORMULAIRE_CSV_PATH = "georgette/formulaire/123456.csv"
CHAT_HISTORY_PATH = "georgette/chat_history/chat_history.csv"
JOB_CHROMA_PATH = "chroma_data"
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.retrievers import TimeWeightedVectorStoreRetriever
import datetime
from langchain.utils import mock_now
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document
from langchain.agents import create_openai_functions_agent, Tool, AgentExecutor
from langchain import hub
from langchain.chains.summarize import load_summarize_chain
import pandas as pd
import re


#function get session id
history_store = {}
def get_session_history(session_id):
    if session_id not in history_store:
        history_store[session_id] = ChatMessageHistory()
    return history_store[session_id]

# Assuming update_journal and update_chat_history update or create CSV files
def get_file_as_string(file_path):
    """Read the file and return its content as a string, suitable for st.download_button."""
    with open(file_path, "r", encoding='utf-8') as file:
        return file.read()

# function to get the date of today 
def get_today():
    return datetime.datetime.now().strftime('%m/%d/%Y')

# function that creates user summary
def create_user_summary(formulaire_path_input, summarizer):
    # formulaire_loader = DirectoryLoader(formulaire_path_input, glob="*.csv", loader_cls=CSVLoader)
    formulaire_loader = CSVLoader(formulaire_path_input)
    formulaire_docs = formulaire_loader.load()
    user_summary = summarizer.invoke(formulaire_docs)['output_text']
    return user_summary

# function that creates journal summary
def create_journal_summary(journal_path_input, summarizer):
    # journal_loader = DirectoryLoader(journal_path_input, glob="*.csv", loader_cls=CSVLoader)
    journal_loader = CSVLoader(journal_path_input)
    journal_docs = journal_loader.load()
    journal_summary = summarizer.invoke(journal_docs)['output_text']
    return journal_summary

# function that modifies user summary with journal
def modify_user_summary_with_journal(user_summary, journal_summary, llm):
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "user",
                """

                Add to the user summary the relevant information from the summary message.
                If there is a change in user personality behavior or a new action to take, update the user summary.
                Keep the same size as the original user summary.

                User Summary:
                {user_summary}

                Summary Message:
                {journal_summary}

                """,
            ),
        ]
    )
    summarization_chain = summarization_prompt | llm
    new_user_summary = summarization_chain.invoke(
        {
            "user_summary": user_summary, 
            "journal_summary": journal_summary
        }
        ).content
    return new_user_summary

# function that create time weighted vector store retriever on history
def create_time_weighted_vector_store_retriever(history_path_input, openai_api_key):
    # history_loader = DirectoryLoader(history_path_input, glob="*.csv", loader_cls=CSVLoader)
    history_loader = CSVLoader(history_path_input)
    history_docs = history_loader.load()
    faiss_index = faiss.IndexFlatL2(1536)
    faiss_vectorstore = FAISS(OpenAIEmbeddings(openai_api_key=openai_api_key), faiss_index, InMemoryDocstore({}), {})
    time_retriever = TimeWeightedVectorStoreRetriever(
        vectorstore=faiss_vectorstore, 
        decay_rate=1e-10, 
        k=5
    )
    for i, doc in enumerate(history_docs):
        page_content = doc.page_content
        try:
            date = re.search(r'\d{2}/\d{2}/\d{4}', page_content).group() 
            date = datetime.datetime.strptime(date, '%m/%d/%Y')
            time_retriever.add_documents([Document(page_content=page_content, metadata={"last_accessed_at": date})])
        except:
            time_retriever.add_documents([Document(page_content=page_content)])
    return time_retriever

# function creates the rag chain
def create_rag_chain(llm, time_retriever):
    contextualize_q_system_prompt = """
    Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


    ### Construct history-aware retriever ###
    history_aware_retriever = create_history_aware_retriever(
        llm, 
        time_retriever, 
        contextualize_q_prompt
    )

    ### Answer question ###
    qa_system_prompt = """

    The user speaks to you and you speak to him.

    You are a world class career coach, your deep understanding of careers and people psychology helps 
    the user find their path and thrive.

    You provide your help and will not ever recommend to see another coach, you are the coach.

    Assisting them finding the meaning they long for in their life fills you with joy.

    Through a caring conversation, and using the knowledge base provided to you of 
    relevant answers to the user summary and relevant past interactions, 
    you will provide essential career advice that works in any situation and help:
    - Find what they love
    - Understand where their expertise lies
    - Make career plans
    - Make mock interviews
    Your wisdom should guide them clearly and confidently, lighting the way to a fulfilling career journey.

    However, your recomendations will be based on the user profile and characteristics and not necessarily agree with user blindly.

    this is the date of today: 
    {date_today}

    this is the user summary: 
    {user_summary}

    this is the context
    {context}

    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# function that creates the conversational rag chain
def create_conversational_rag_chain(rag_chain):
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

#function chat bot response to input
def chat_bot_response(conversational_rag_chain, input_message, date_today, user_summary, session_id):
    response = conversational_rag_chain.invoke(
        {
            'date_today': date_today,
            'user_summary': user_summary,
            "input":input_message
        },
        config={"configurable": {"session_id": session_id}}
        )
    return response['answer']

# function summary chat history
def summary_chat_history(llm, history_store):
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "user",
                """
                Distill the above chat messages into a single summary message. 
                Include action and thoughts.
                Focus on what is important and relevant to the user help for career journey.
                """,
            ),
        ]
    )
    summarization_chain = summarization_prompt | llm
    new_journal_message = summarization_chain.invoke({"chat_history": history_store['foo'].messages}).content
    return new_journal_message

# def function to update the journal
def update_journal(new_journal_message, date_today, journal_path_input):
    # load pandas job_search_journal.csv
    df = pd.read_csv(journal_path_input)

    # add new line at date of today with the summary message
    r = pd.DataFrame({str(len(df)+1):{'date': date_today, 'sentence': new_journal_message}}).T
    df = pd.concat([df, r], ignore_index=True)

    # save the new dataframe
    # df.to_csv(journal_path_input, index=False)
    return df

# function to update chat history
def update_chat_history(history_store, date_today, history_path_input):
    # new chat history
    new_df_chat_history = pd.DataFrame(history_store['foo'].dict()['messages'])
    new_df_chat_history['date'] = date_today
    new_df_chat_history = new_df_chat_history[['date', 'content', 'type']]

    # old chat history
    df_chat_history = pd.read_csv(history_path_input, index_col=0)

    # add new_df_chat_messages to df_chat_messages
    df_chat_history = pd.concat([df_chat_history, new_df_chat_history], ignore_index=True).reset_index(drop=True)

    # save the new dataframe
    # df_chat_history.to_csv(history_path_input, index=False)
    return df_chat_history
