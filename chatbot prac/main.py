import streamlit as st
from streamlit.delta_generator import DeltaGenerator as _DeltaGenerator
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.runtime import Runtime, RuntimeConfig, RuntimeState
from streamlit.proto.BackMsg_pb2 import BackMsg
from google.protobuf.internal import builder as _builder


from langchain_helpr import get_qa_chain, create_vector_db

st.title("Codebasics Q&A 🌱")
btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])

