import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ----------------- Page Configuration & Styling -----------------
st.set_page_config(
    page_title="Soulslike Compendium",
    page_icon="⚔️",
    layout="centered"
)

# Custom CSS to match the dark theme and UI elements
st.markdown("""
    <style>
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Header styling */
    .custom-header {
        text-align: center;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
    }
    .custom-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    .custom-subtitle {
        text-align: center;
        color: #8b949e;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    
    /* Chat message containers */
    .user-container {
        display: flex;
        align-items: center;
        background-color: #161b22;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 16px;
        gap: 12px;
    }
    .bot-container {
        display: flex;
        align-items: flex-start;
        background-color: transparent;
        border-radius: 8px;
        padding: 4px 0px;
        margin-bottom: 20px;
        gap: 12px;
    }
    .avatar-user {
        background-color: #f87171;
        color: white;
        border-radius: 6px;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        flex-shrink: 0;
    }
    .avatar-bot {
        background-color: #e06c75;
        color: #ffffff;
        border-radius: 6px;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        flex-shrink: 0;
    }
    .user-text {
        color: #e6edf3;
        font-size: 0.95rem;
    }
    .bot-text {
        color: #c9d1d9;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- Setup RAG Pipeline -----------------
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

@st.cache_resource
def get_rag_chain():
    # Soulslike knowledge base documents
    sample_docs = [
        Document(
            page_content=(
                "Soulslike is a subgenre of action role-playing games originated by FromSoftware's Demon's Souls "
                "and the Dark Souls series, directed by Hidetaka Miyazaki. It is characterized by high difficulty, "
                "stamina-managed combat, punishing death loops, interconnected world design, and environmental storytelling."
            ),
            metadata={"topic": "Genre Definition"}
        ),
        Document(
            page_content=(
                "Bonfires and Checkpoints: Resting at a checkpoint (such as a Bonfire, Site of Grace, or Lamp) "
                "restores the player's health and healing items (like Estus Flasks) but causes most defeated "
                "enemies in the area to respawn."
            ),
            metadata={"topic": "Checkpoints & Healing"}
        ),
        Document(
            page_content=(
                "Death and Currency: Defeating enemies yields currency (Souls, Runes, Ergo, Blood Echoes) "
                "used for leveling up attributes and buying items. Upon dying, all held currency is dropped at the "
                "spot of death, and the player gets exactly one chance to retrieve it before it is lost forever."
            ),
            metadata={"topic": "Death Mechanics"}
        ),
        Document(
            page_content=(
                "Combat Philosophy: Combat relies on deliberate stamina management, animation commitments (i-frames on dodge "
                "rolls, parrying windows), reading enemy telegraphs, and intense multi-phase boss encounters."
            ),
            metadata={"topic": "Combat Mechanics"}
        ),
        Document(
            page_content=(
                "Notable Soulslike Titles: FromSoftware titles include Demon's Souls, Dark Souls 1-3, Bloodborne, "
                "Sekiro: Shadows Die Twice, and Elden Ring. Non-FromSoftware examples include Nioh, Lies of P, "
                "Lords of the Fallen, Hollow Knight, and The Surge."
            ),
            metadata={"topic": "Notable Games"}
        )
    ]

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    
    vectorstore = FAISS.from_documents(sample_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for the Soulslike Database.
Answer the question accurately based on the provided context.

Context:
{context}

Question: {question}

Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ----------------- UI Layout & Chat Flow -----------------
st.markdown("""
    <div class="custom-header">
        <h1>⚔️ Soulslike Database</h1>
    </div>
    <div class="custom-subtitle">
        Ask me anything about Soulslike mechanics, boss design, lore delivery, and titles!
    </div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
            <div class="user-container">
                <div class="avatar-user">👤</div>
                <div class="user-text">{msg['content']}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="bot-container">
                <div class="avatar-bot">🔥</div>
                <div class="bot-text">{msg['content']}</div>
            </div>
        """, unsafe_allow_html=True)

# User input field
user_input = st.chat_input("E.g., What happens to souls when a player dies?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f"""
        <div class="user-container">
            <div class="avatar-user">👤</div>
            <div class="user-text">{user_input}</div>
        </div>
    """, unsafe_allow_html=True)

    if not GOOGLE_API_KEY:
        response_text = "Please set the `GEMINI_API_KEY` environment variable to run queries."
    else:
        try:
            rag_chain = get_rag_chain()
            response_text = rag_chain.invoke(user_input)
        except Exception as e:
            response_text = f"Error retrieving answer: {e}"

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.markdown(f"""
        <div class="bot-container">
            <div class="avatar-bot">🔥</div>
            <div class="bot-text">{response_text}</div>
        </div>
    """, unsafe_allow_html=True)