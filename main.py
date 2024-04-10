import os
# from dotenv import load_dotenv, find_dotenv
# from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
# from langchain.embeddings import GooglePalmEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
# from langchain.llms import AzureOpenAI
# from langchain.document_loaders import DirectoryLoader,PyPDFLoader
# from langchain.document_loaders import UnstructuredExcelLoader
# from langchain.vectorstores import DocArrayInMemorySearch
from langchain.memory import ConversationBufferMemory
# from IPython.display import display, Markdown
# import pandas as pd
# import gradio as gr
# from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate
# from langchain.vectorstores import Chroma
from langchain.agents.tools import Tool
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
# from langchain import OpenAI, VectorDBQA
# from langchain.chains.router import MultiRetrievalQAChain
import streamlit as st
import pandas as pd
# from langchain.document_loaders import UnstructuredPDFLoader
# _ = load_dotenv(find_dotenv())

# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     AIMessagePromptTemplate,
#     HumanMessagePromptTemplate,
# )
# from langchain.schema import (
#     AIMessage,
#     HumanMessage,
#     SystemMessage
# )

import os
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.embeddings import GooglePalmEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI, ChatAnthropic
from langchain.llms import AzureOpenAI
from langchain.document_loaders import DirectoryLoader,PyPDFLoader
# from langchain.document_loaders import UnstructuredExcelLoader
# from langchain.vectorstores import DocArrayInMemorySearch
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate
# from langchain.vectorstores import Chroma
# from langchain.agents.tools import Tool
# from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
# from langchain import OpenAI, VectorDBQA
# from langchain.chains.router import MultiRetrievalQAChain
import streamlit as st
import pandas as pd
from tqdm import tqdm
import utils
# from langchain.document_loaders import UnstructuredPDFLoader

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from PIL import Image

utils.setup_page()

session = st.session_state
utils.setup_session(session)

openai_llm, embeddings, anthropic_llm = utils.setup_llm()

all_documents = {
    "Payment Services Act"                              :   {"data":"data/Payment Services Act 2019 - Notice on regulatory returns.pdf","index":"Payment Services Act 2019 - Notice on regulatory returns_index"},
    "Bank of Montreal (BMO)"            :   {"data":"./data/bmo_ar2022_removed.pdf","index":"bmo_ar2022 (2)_index"},
    "Versa Bank (VB)"                   :   {"data":"./data/Versa bank","index":"VBAR_index"},
    "Home Bancorp (HB)"                 :   {"data":"./data/Home Bancorp 2022 Annual Report.pdf"},
    "National Bank of Canada (NBC)"     :   {"data":"./data/NATIONAL BANK OF CANADA_ 2022 Annual Report (1).pdf","index":"NATIONAL BANK OF CANADA_ 2022 Annual Report (1)_index"},
        "OCBC"                          :   {"data":"./data/2023-annual-report-OCBC.pdf","index": "2023-annual-report-OCBC_index"},
    }

institutes = all_documents.copy()
del institutes["Payment Services Act"]

with st.sidebar:
    # l,r = st.columns([1,1.])
    st.markdown("# Reg Reporting Assistance")
    # r.image(Image.open('osfi_logo.png'),width=40)
    institute = st.selectbox(label="Institute",options=institutes,label_visibility="hidden",disabled=session.analyze_disabled)

def analyse():
    with st.sidebar:
        with st.spinner("Loading documents..."):
            session.analyze_disabled = True
            session.institute = institute
            session.docs = {
            f"{session.institute} Annual Report"        :   utils.load_doc(all_documents[session.institute]["data"]),
            "Payment Services Act"   :   utils.load_doc(all_documents["BCAR"]["data"]),
            }                            
            session.input_disabled = False
            session.transcript.append(["assistant","How can I help you today?"])

with st.sidebar:
    analyze_button = st.button("Load Documents",use_container_width=True,disabled=session.analyze_disabled,on_click=analyse)                           

# l,r = st.columns([2,1])
user_input = st.chat_input("Query",disabled=session.input_disabled)

if user_input:
    session.transcript.append(["user",user_input])
    with st.spinner("Processing..."):
        bot_details, bot_output = utils.compare_answer(anthropic_llm,anthropic_llm,session,user_input,session.docs)
    session.transcript.append(["system",bot_details,user_input])
    session.transcript.append(["assistant",bot_output])


if len(session.transcript)>0:
    for message in session.transcript:
        if message[0]=="system":
            with st.sidebar:
                with st.expander(message[2]):
                    st.write(message[1])
        else:
            st.chat_message(message[0]).write(message[1])

