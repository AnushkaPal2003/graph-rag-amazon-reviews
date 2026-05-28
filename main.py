
import os
from dotenv import load_dotenv

from data_loader import load_data, compute_phone_stats, compute_brand_stats
from neo4j_setup import (
    connect_to_neo4j,
    clear_database,
    create_brand_nodes,
    create_phone_nodes,
    create_review_nodes,
    create_relationships,
    verify_graph,
)
from embeddings  import store_phone_embeddings, store_review_embeddings
from rag_engine  import get_ai_client, answer_question

load_dotenv()


# FUNCTION 1: First time setup

def setup_everything():
    print("\n First time setup running...")

    # Load and clean the Kaggle dataset
    df          = load_data()
    phone_stats = compute_phone_stats(df)
    brand_stats = compute_brand_stats(df)

    # Connect to Neo4j
    driver = connect_to_neo4j()
    if not driver:
        return None

    # Build the knowledge graph
    clear_database(driver)
    create_brand_nodes(driver, df)
    create_phone_nodes(driver, phone_stats)
    create_review_nodes(driver, df)
    create_relationships(driver, df)
    verify_graph(driver)

    # Create embeddings
    print("\n Creating embeddings for semantic search...")
    store_phone_embeddings(driver)
    
    store_review_embeddings(driver, limit=2000)

    print("\n Setup complete — graph + embeddings ready!")
    return driver


# FUNCTION 2: Quick stats

def show_quick_stats(driver):

    print("\n" + "─" * 55)
    print("   AMAZON PHONE REVIEWS — QUICK STATS")
    print("─" * 55)

    with driver.session() as session:

        result = session.run("""
            MATCH (p:Phone)
            RETURN p.name AS name, p.avg_rating AS rating
            ORDER BY p.avg_rating DESC LIMIT 1
        """)
        best = result.single()
        if best:
            print(f" Best Phone     : {best['name']} ({best['rating']}/5)")

        result = session.run("MATCH (r:Review) RETURN count(r) AS total")
        total  = result.single()
        if total:
            print(f" Total Reviews  : {total['total']}")

        result = session.run("MATCH (p:Phone) RETURN count(p) AS total")
        phones = result.single()
        if phones:
            print(f" Total Phones   : {phones['total']}")

        # Check if embeddings exist
        result = session.run("""
            MATCH (p:Phone)
            WHERE p.embedding IS NOT NULL
            RETURN count(p) AS total
        """)
        emb = result.single()
        if emb and emb["total"] > 0:
            print(f" Embedded Phones: {emb['total']} (semantic search ON ✅)")
        else:
            print(" Embeddings     : Not created yet (semantic search OFF)")

    print("─" * 55)



# Sample questions

SAMPLE_QUESTIONS = [
    "Which is the best phone to buy?",
    "Compare Apple vs Samsung",
    "Any phone with a great camera?",
    "Tell me about heating issues in phones",
    "Which phone has the most reviews?",
    "What do customers say about OnePlus?",
    "Which brand has the best rating?",
    "What is the cheapest good phone?",
]


def show_sample_questions():
    print("\n Sample questions (type number or write your own):")
    for i, q in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"   {i}. {q}")
    print("\n   Type 'exit' to quit\n")


# FUNCTION 3: Chat loop

def chat_loop(driver, ai_client):

    print("\n" + "=" * 55)
    print("  GRAPH RAG PHONE ASSISTANT")
    print("  Keyword Search + Semantic Search + GPT")
    print("=" * 55)

    show_quick_stats(driver)
    show_sample_questions()

    while True:

        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\n Goodbye!")
            break

        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(SAMPLE_QUESTIONS):
                user_input = SAMPLE_QUESTIONS[idx]
                print(f"   → Asking: {user_input}")

        # Run the full Graph RAG pipeline
        answer = answer_question(driver, ai_client, user_input)

        print(f"\n Answer:\n{answer}")
        print("─" * 50)

# MAIN

if __name__ == "__main__":

    print("=" * 55)
    print("  GRAPH RAG — Amazon Phone Reviews")
    print("  Neo4j + Embeddings + OpenAI GPT")
    print("=" * 55)

    if not os.path.exists(".env"):
        print("\n .env file not found!")
        print("   Copy .env.example to .env and fill your keys.")
        exit(1)

    # Connect to Neo4j
    driver = connect_to_neo4j()
    if not driver:
        exit(1)

    # Check if graph already has data
    with driver.session() as session:
        result     = session.run("MATCH (n) RETURN count(n) AS count")
        node_count = result.single()["count"]

  
    if node_count == 0 or node_count > 4000:
        print("\n Database requires alignment. Running setup cycle...")
        driver = setup_everything()
        if not driver:
            exit(1)
    else:
        print(f"\n Graph exists ({node_count} nodes). Skipping setup.")

    # OpenAI client
    ai_client = get_ai_client()
    if not ai_client:
        exit(1)

    
    chat_loop(driver, ai_client)

    driver.close()