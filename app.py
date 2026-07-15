import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain      
from langchain.memory import ConversationBufferMemory          
from langchain.prompts import PromptTemplate  

st.set_page_config(page_title="AI Chatbot", layout="wide")
st.title("AI Support Chatbot")

st.markdown("""
<style>
.chat-user { background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin: 6px 0; text-align: right; }
.chat-bot  { background-color: #F1F0F0; padding: 10px; border-radius: 10px; margin: 6px 0; text-align: left; }
</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def load_vectorstore():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

@st.cache_resource
def load_llm():
    return ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=512
    )

@st.cache_resource
def load_qa_chain(_vectorstore, _llm):
    retriever = _vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    template = """You are an AI assistant. Answer ONLY the question asked.
Use only the most relevant part of the context. Answer in 2-3 complete sentences.
Context: {context}
Question: {question}
Answer:"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return ConversationalRetrievalChain.from_llm(
        llm=_llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

vectorstore = load_vectorstore()
llm         = load_llm()
qa_chain    = load_qa_chain(vectorstore, llm)

st.sidebar.title("⚙️ Settings")
st.sidebar.write("**Model:** Llama 3.3 70B (Groq)")
st.sidebar.write("**Embeddings:** MiniLM-L6-v2")
st.sidebar.write("**Vector Store:** FAISS")
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    qa_chain.memory.clear()
    st.rerun()

query = st.chat_input("Ask your question...")
if query:
    try:
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"question": query})
            answer = result.get("answer", "Sorry, I couldn't find an answer.")
    except Exception as e:
        answer = f"Error: {str(e)}"
    st.session_state.chat_history.append(("user", query))
    st.session_state.chat_history.append(("bot", answer))

for role, message in st.session_state.chat_history:
    css_class = "chat-user" if role == "user" else "chat-bot"
    st.markdown(f"<div class='{css_class}'>{message}</div>", unsafe_allow_html=True)
