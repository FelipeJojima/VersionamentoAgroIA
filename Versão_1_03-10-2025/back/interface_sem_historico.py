import streamlit as st
from model import search


st.set_page_config(page_title="BuscaAgro", layout="centered")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Faça sua pergunta"}]
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
pergunta = st.chat_input("Faça sua pergunta.")

if pergunta:
    st.session_state.messages.append({"role": "user", "content": pergunta})
    st.chat_message("user").write(pergunta)
    response = search(pergunta)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
