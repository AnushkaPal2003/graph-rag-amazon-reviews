

import streamlit as st
import os
from dotenv import load_dotenv

from neo4j_setup import connect_to_neo4j
from rag_engine  import get_ai_client, answer_question

load_dotenv()


# Page config

st.set_page_config(
    page_title = "Phone Advisor — Graph RAG",
    page_icon  = "📱",
    layout     = "centered"
)


# Initialize connections — only once using session state
# session_state = remembers values across reruns

if "driver" not in st.session_state:
    st.session_state.driver = connect_to_neo4j()

if "ai_client" not in st.session_state:
    st.session_state.ai_client = get_ai_client()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Header


st.title("Phone Advisor")
st.caption("Powered by Graph RAG — Neo4j + Azure OpenAI + Embeddings")
st.divider()

# Sidebar — project info
with st.sidebar:
    st.header("About this project")

    st.markdown("""
    This is a **Graph RAG** system that answers
    questions about phones using real Amazon reviews.

    **How it works:**
    1. Your question goes to an LLM Router
    2. Neo4j Knowledge Graph is searched
    3. Keyword + Semantic search runs
    4. Azure OpenAI generates the answer
    """)

    st.divider()

    st.header("Graph Stats")
    if st.session_state.driver:
        try:
            with st.session_state.driver.session() as session:
                r = session.run("MATCH (b:Brand) RETURN count(b) AS c").single()
                st.metric("Brands",  r["c"] if r else 0)

                r = session.run("MATCH (p:Phone) RETURN count(p) AS c").single()
                st.metric("Phones",  r["c"] if r else 0)

                r = session.run("MATCH (r:Review) RETURN count(r) AS c").single()
                st.metric("Reviews", r["c"] if r else 0)

                r = session.run("""
                    MATCH (p:Phone)
                    WHERE p.embedding IS NOT NULL
                    RETURN count(p) AS c
                """).single()
                embedded = r["c"] if r else 0
                st.metric("Embedded Phones", embedded)

                if embedded > 0:
                    st.success("Semantic search ON")
                else:
                    st.warning("Semantic search OFF")

        except Exception as e:
            st.error(f"Neo4j error: {e}")
    else:
        st.error("Neo4j not connected")

    st.divider()

    st.header("Sample Questions")
    sample_questions = [
        "Which is the best phone to buy?",
        "Compare Apple vs Samsung",
        "Any phone with a great camera?",
        "Heating issues in phones?",
        "Which brand has the best rating?",
        "Cheapest good phone?",
    ]

    for q in sample_questions:
        # Button for each sample question
        if st.button(q, use_container_width=True):
            st.session_state.selected_question = q


# Chat History — show previous questions and answers
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        st.write(chat["answer"])


# Input — text box at bottom of page


# Check if a sample question was clicked
default_input = ""
if "selected_question" in st.session_state:
    default_input = st.session_state.selected_question
    del st.session_state.selected_question

# Chat input box
user_input = st.chat_input("Ask anything about phones...")

# Use either typed input or clicked sample question
question = user_input or default_input

# Run RAG pipeline when question is submitted


if question:

    # Show user message immediately
    with st.chat_message("user"):
        st.write(question)

    # Show spinner while generating answer
    with st.chat_message("assistant"):
        with st.spinner("Searching Knowledge Graph..."):

            if not st.session_state.driver:
                answer = "ERROR: Neo4j not connected. Check your .env file."

            elif not st.session_state.ai_client:
                answer = "ERROR: Azure OpenAI not connected. Check your .env file."

            else:
                # Run the full Graph RAG pipeline
                answer = answer_question(
                    st.session_state.driver,
                    st.session_state.ai_client,
                    question
                )

        st.write(answer)

    # Save to chat history
    st.session_state.chat_history.append({
        "question": question,
        "answer"  : answer
    })

# Clear chat button


if st.session_state.chat_history:
    if st.button("Clear Chat", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()