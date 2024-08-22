
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from chatgpt import openai_embeddings, llm_chat
from history import History

company_name = "Crunchbase"
st.set_page_config(
    page_title=f"{company_name} Startup GPT",
    page_icon="🚀",
    layout="wide"
)

pages = PyPDFLoader(f"{company_name.lower()}.pdf").load_and_split()


def query_dataset(query: str, k: int):

    # find top 10 most similar passages for query
    new_db = FAISS.from_documents(pages, openai_embeddings)
    documents = new_db.similarity_search(query, k=k)

    # load documents into a history and using LLM to get answer to query
    history = History()
    for document in documents:
        print(document.page_content)
        history.system(document.page_content)

    history.user(query)

    answer = llm_chat(history)
    return answer


st.title(f"{company_name} Startup GPT")

# check for messages in session and create if not exists
if "history" not in st.session_state.keys():
    st.session_state.history = History()

    st.session_state.history.system(f"""You are a very kindly and friendly startup assistant for {company_name}. You are
    currently having a conversation with a startup person. Answer the questions in a kind and friendly 
    with you being the expert for {company_name} to answer any questions about startups and investing.""")
    st.session_state.history.assistant("Hello there, how can I help you? 📙")


# Display all messages
for message in st.session_state.history.logs:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.history.user(user_prompt)
    with st.chat_message("user"):
        st.write(user_prompt)

if st.session_state.history.logs[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            chat = llm_chat(st.session_state.history)
            st.write(chat)
    st.session_state.history.assistant(chat)
