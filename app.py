import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Support Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("AI Support Chatbot")

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.chat-user {
    background-color: #DCF8C6;
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
    text-align: right;
}

.chat-bot {
    background-color: #F1F0F0;
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def clear_chat():
    st.session_state.chat_history = []

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Settings")
st.sidebar.write("Model: FLAN-T5 Base")
st.sidebar.write("Embeddings: MiniLM-L6-v2")
st.sidebar.write("Vector DB: FAISS")

if st.sidebar.button("🗑️ Clear Chat"):
    clear_chat()
    st.rerun()

# =========================
# LOAD VECTOR STORE
# =========================
@st.cache_resource
def load_vectorstore():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        ".",
        embedding,
        index_name="index",
        allow_dangerous_deserialization=True
    )

    return vectorstore

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model

# =========================
# GENERATION FUNCTION (IMPROVED)
# =========================
def generate_answer(tokenizer, model, context, question):

    prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the context below.

RULES:
- Use only provided context
- Do NOT copy text directly
- Explain clearly in 2–5 sentences
- If not in context, say: "I don't know based on the provided information."

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2
        )

    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean possible prompt echo
    answer = raw.replace(prompt, "").strip()

    return answer

# =========================
# LOAD RESOURCES
# =========================
try:
    vectorstore = load_vectorstore()
    tokenizer, model = load_model()

except Exception as e:
    st.error(f"Startup Error: {e}")
    st.stop()

# =========================
# CHAT INPUT
# =========================
query = st.chat_input("Ask your question...")

if query:
    try:
        with st.spinner("Thinking..."):

            #  IMPROVED RETRIEVAL (MMR)
            docs = vectorstore.max_marginal_relevance_search(
                query,
                k=3,
                fetch_k=10
            )

            context = "\n\n".join([doc.page_content for doc in docs])

            answer = generate_answer(tokenizer, model, context, query)

    except Exception as e:
        answer = f"Error: {str(e)}"

    st.session_state.chat_history.append(("user", query))
    st.session_state.chat_history.append(("bot", answer))

# =========================
# DISPLAY CHAT
# =========================
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(
            f"<div class='chat-user'>{message}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='chat-bot'>{message}</div>",
            unsafe_allow_html=True
        )
