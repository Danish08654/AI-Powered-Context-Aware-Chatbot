import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

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

def search(vectorstore, query, k=2):
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)

def ask(query, context, history):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    messages = [{"role": "system", "content": """You are an AI assistant. 
Answer ONLY using the context provided. Answer in 2-3 sentences.
If the answer is not in the context, say 'I couldn't find that in the documents.'"""}]
    for role, msg in history[-6:]:
        messages.append({"role": "user" if role == "user" else "assistant", "content": msg})
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2,
        max_tokens=512
    ).choices[0].message.content.strip()

vectorstore = load_vectorstore()

st.sidebar.title("⚙️ Settings")
st.sidebar.write("**Model:** Llama 3.3 70B (Groq)")
st.sidebar.write("**Embeddings:** MiniLM-L6-v2")
st.sidebar.write("**Vector Store:** FAISS")
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

query = st.chat_input("Ask your question...")
if query:
    try:
        with st.spinner("Thinking..."):
            context = search(vectorstore, query)
            answer  = ask(query, context, st.session_state.chat_history)
    except Exception as e:
        answer = f"Error: {str(e)}"
    st.session_state.chat_history.append(("user", query))
    st.session_state.chat_history.append(("bot", answer))

for role, message in st.session_state.chat_history:
    css_class = "chat-user" if role == "user" else "chat-bot"
    st.markdown(f"<div class='{css_class}'>{message}</div>", unsafe_allow_html=True)
