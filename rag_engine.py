import os
from openai import AzureOpenAI
from neo4j  import GraphDatabase
from dotenv import load_dotenv

from graph_query import build_context_for_question
from embeddings  import build_semantic_context

load_dotenv()


# FUNCTION 1: OpenAI client

def get_ai_client():
    """
    Creates and returns the OpenAI client.
    Reads API key from .env file.
    """

    api_key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    if not api_key:
        print(" AZURE_OPENAI_KEY missing in .env!")
        return None

    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-02-01",
        azure_endpoint=endpoint
    )
    print(" Azure OpenAI client ready!")
    return client



# FUNCTION 2: LLM Router — decide what type of query this is

def decide_query_type(ai_client, question):
    """
    Sends the question to GPT and asks it to classify
    what kind of data we need from Neo4j.

    This is called an LLM ROUTER — instead of writing
    20 if/else conditions, we let GPT decide.

    Returns one label:
    TOP_PHONES / BRAND_QUERY / PHONE_REVIEWS /
    BRAND_COMPARISON / GENERAL
    """

    system_message = (
        "You are a strict backend API routing agent. Your ONLY job is to classify the user's intent. "
        "You must output exactly ONE word from the allowed list, with absolutely no formatting, "
        "no punctuation, no conversational filler, and no explanations.\n"
        "Allowed labels:\n"
        "- TOP_PHONES\n"
        "- BRAND_QUERY\n"
        "- PHONE_REVIEWS\n"
        "- BRAND_COMPARISON\n"
        "- GENERAL"
    )

    user_prompt = f"""Classify this user question: "{question}" """

    response = ai_client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),  
        messages=[
            # CHANGED: Separated systemic constraints from user payload
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
    )
    
   
    return response.choices[0].message.content.strip().upper()



# FUNCTION 3: Hybrid retrieval — keyword + semantic


def fetch_context(driver, question, query_type="GENERAL", use_semantic=True):
    """
    Fetches relevant data from Neo4j using TWO methods:

    1. KEYWORD retrieval  — fast, exact matches
    2. SEMANTIC retrieval — slow, meaning-based matches

    Combines both results for the best possible context.
    This is called HYBRID SEARCH — used in production RAG systems.
    """

    context_parts = []

    # Method 1: Keyword-based retrieval (always runs)
   
    keyword_context = build_context_for_question(driver, question, query_type=query_type)
    if keyword_context.strip():
        context_parts.append("--- Keyword Search Results ---")
        context_parts.append(keyword_context)

    # Method 2: Semantic retrieval (runs if embeddings exist)
    if use_semantic:
        try:
            semantic_context = build_semantic_context(driver, question)
            if semantic_context.strip():
                context_parts.append("\n--- Semantic Search Results ---")
                context_parts.append(semantic_context)
        except Exception as e:
            # If embeddings not set up yet, skip gracefully
            print(f"Semantic search skipped: {e}")

    return "\n".join(context_parts)

# FUNCTION 4: Generate final answer with GPT

def generate_answer(ai_client, question, context):
    """
    Sends context (from Neo4j) + question to GPT.
    GPT reads the data and generates a helpful answer.

    This is the GENERATION step of RAG.
    """

    model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    system_prompt = """You are a helpful phone shopping assistant.
You have access to real Amazon customer review data.
Answer ONLY based on the data provided to you.
Be specific — mention actual phone names, ratings, numbers.
Keep the answer friendly, clear, and helpful."""

    user_prompt = f"""Here is the relevant data from our database:

{context}

---
Customer Question: {question}

Give a clear helpful recommendation based on the data above."""

    response = ai_client.chat.completions.create(
        model    = model,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens  = 512,
        temperature = 0.3,
    )

    answer = response.choices[0].message.content

    # Show token usage to track budget
    usage = response.usage
    print(f" Tokens — prompt: {usage.prompt_tokens} | "
          f"response: {usage.completion_tokens} | "
          f"total: {usage.total_tokens}")

    return answer

# FUNCTION 5: Complete Graph RAG pipeline

def answer_question(driver, ai_client, question):
    """
    THE COMPLETE GRAPH RAG PIPELINE:

    Question
        ↓
    [ROUTER]    GPT decides query type
        ↓
    [RETRIEVE]  Keyword search + Semantic search from Neo4j
        ↓
    [GENERATE]  GPT reads context and answers
        ↓
    Answer
    """

    print(f"\n Question: '{question}'")

    # Step 1: Route — what kind of query is this?
    query_type = decide_query_type(ai_client, question)
    print(f" Query type: {query_type}")

    # Step 2: Retrieve — fetch from Neo4j (hybrid search)
    # CHANGED: Added query_type mapping keyword transmission parameter link here
    context = fetch_context(driver, question, query_type=query_type, use_semantic=True)
    print(f" Context fetched ({len(context)} characters)")

    if not context.strip():
        return "Sorry, no relevant data found. Try asking about phones, brands, or reviews."

    # Step 3: Generate — GPT reads context and answers
    print(" Generating answer...")
    answer = generate_answer(ai_client, question, context)

    return answer


# TEST — python rag_engine.py

if __name__ == "__main__":

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD"))
    )

    ai_client = get_ai_client()

    test_questions = [
        "Which is the best phone to buy?",
        "Any phone with good camera?",   
        "heating problems in phones",   
    ]

    for q in test_questions:
        print("\n" + "="*50)
        answer = answer_question(driver, ai_client, q)
        print(f"\n Answer:\n{answer}")

    driver.close()