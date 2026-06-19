import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from transformers import pipeline

# =========================
# UI CONFIG
# =========================
st.set_page_config(page_title="AI Chatbot", layout="wide")
st.title("🤖 AI Support Chatbot")

# =========================
# CSS
# =========================
st.markdown("""
<style>
.chat-user {
    background-color: #DCF8C6;
    padding: 10px;
    border-radius: 10px;
    margin: 6px 0;
    text-align: right;
}
.chat-bot {
    background-color: #F1F0F0;
    padding: 10px;
    border-radius: 10px;
    margin: 6px 0;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = None


# =========================
# LOAD VECTOR DB
# =========================
@st.cache_resource
def load_vectorstore():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # IMPORTANT: use correct folder (current directory)
    vectorstore = FAISS.load_local(
        ".",
        embedding,
        allow_dangerous_deserialization=True
    )

    return vectorstore


# =========================
# LOAD LLM (FIXED PIPELINE)
# =========================
@st.cache_resource
def load_llm():
    pipe = pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7
    )

    return HuggingFacePipeline(pipeline=pipe)


# =========================
# QA CHAIN
# =========================
@st.cache_resource
def load_qa_chain(vectorstore, llm):

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    st.session_state.memory = memory

    prompt = PromptTemplate(
        template="""
You are a helpful AI assistant.

Use ONLY the context below.
If unknown, say "I don't know".

Context:
{context}

Question:
{question}

Answer in 2-3 sentences:
""",
        input_variables=["context", "question"]
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False
    )

    return chain


# =========================
# INIT
# =========================
vectorstore = load_vectorstore()
llm = load_llm()
qa_chain = load_qa_chain(vectorstore, llm)


# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Settings")
st.sidebar.write("Model: FLAN-T5 Base")
st.sidebar.write("Embeddings: MiniLM-L6-v2")

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    if st.session_state.memory:
        st.session_state.memory.clear()
    st.rerun()


# =========================
# CHAT INPUT
# =========================
query = st.chat_input("Ask your question...")

if query:
    with st.spinner("Thinking..."):
        try:
            result = qa_chain.invoke({"question": query})
            answer = result["answer"]
        except Exception as e:
            answer = f"Error: {str(e)}"

    st.session_state.chat_history.append(("user", query))
    st.session_state.chat_history.append(("bot", answer))


# =========================
# CHAT DISPLAY
# =========================
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"<div class='chat-user'>{msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bot'>{msg}</div>", unsafe_allow_html=True)
