import streamlit as st
import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
from script import Conversation


st.set_page_config(page_title="PageMind", page_icon="", layout="centered")
st.title(" PageMind")
st.caption("Ask anything about any webpage")

url = st.text_input(" Enter a URL to get started", placeholder="https://example.com/page")

if url:
    
    if "conversation" not in st.session_state or st.session_state.get("loaded_url") != url:
        with st.spinner(" Scraping and indexing the page…"):
            st.session_state["conversation"] = Conversation(url)
            st.session_state["loaded_url"]   = url
            st.session_state["messages"]     = []  

    if not st.session_state["messages"]:
        with st.chat_message("assistant"):
            st.write("Hello  I've indexed the page. Ask me anything about it!")

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if query := st.chat_input("Ask a question about the page…"):
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = st.session_state["conversation"].chat(query)
            st.write(reply)

        st.session_state["messages"].append({"role": "assistant", "content": reply})
