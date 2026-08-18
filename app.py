import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
    page_title="Soulslike Database",
    page_icon="⚔️",
    layout="centered"
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

SOULSLIKE_KNOWLEDGE = """
Soulslike is a subgenre of action role-playing games originated by FromSoftware's Demon's Souls
and the Dark Souls series, directed by Hidetaka Miyazaki. It is characterized by high difficulty,
stamina-managed combat, punishing death loops, interconnected world design, and environmental storytelling.

Bonfires and Checkpoints: Resting at a checkpoint (such as a Bonfire, Site of Grace, or Lamp)
restores the player's health and healing items but causes most defeated enemies in the area to respawn.

Death and Currency: Defeating enemies yields currency (Souls, Runes, Ergo, Blood Echoes)
used for leveling up attributes and buying items. Upon dying, all held currency is dropped at the
spot of death, and the player gets exactly one chance to retrieve it before it is lost forever.

Combat Philosophy: Combat relies on deliberate stamina management, animation commitments,
dodge-roll i-frames, parrying windows, reading enemy telegraphs, and multi-phase boss encounters.

Notable Soulslike Titles include Demon's Souls, Dark Souls 1-3, Bloodborne,
Sekiro: Shadows Die Twice, Elden Ring, Nioh, Lies of P, Lords of the Fallen,
Hollow Knight, and The Surge.
"""

@st.cache_resource
def get_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a Soulslike Database assistant.

Use ONLY the provided context.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return chain

st.title("⚔️ Soulslike Database")
st.caption("Ask me anything about Soulslike games!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask a question...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    if not GEMINI_API_KEY:
        answer = "GEMINI_API_KEY not found."
    else:
        try:
            chain = get_chain()

            answer = chain.invoke({
                "context": SOULSLIKE_KNOWLEDGE,
                "question": question
            })

        except Exception as e:
            answer = f"Error: {e}"

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.write(answer)
