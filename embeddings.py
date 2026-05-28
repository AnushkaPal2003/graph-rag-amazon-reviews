import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()


print("Loading embedding model...")
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready!")


# FUNCTION 1: Convert any text into a vector (embedding)
def get_embedding(text):
    # encode() converts text to numpy array
    vector = EMBEDDING_MODEL.encode(text)

    # Convert numpy array to plain Python list
    # Neo4j stores lists, not numpy arrays
    return vector.tolist()


# FUNCTION 2: Calculate similarity between two vectors
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    # Dot product divided by product of magnitudes
    dot_product = np.dot(vec1, vec2)
    magnitude   = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    if magnitude == 0:
        return 0.0

    return float(dot_product / magnitude)


# FUNCTION 3: Store embeddings in Neo4j
def store_phone_embeddings(driver):
    print("\n Creating embeddings for all phones...")

    # Fetch all phones from Neo4j
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Phone)
            RETURN p.name         AS name,
                   p.brand       AS brand,
                   p.avg_rating  AS rating,
                   p.review_count AS reviews,
                   p.avg_price   AS price
        """)
        phones = [dict(record) for record in result]

    total_phones = len(phones)
    print(f"   Found {total_phones} phones to embed...")

    batch_data = []
    batch_size = 100

    for i, phone in enumerate(phones):
        # Create a rich text description of the phone
        description = (
            f"{phone['name']} made by {phone['brand']}. "
            f"Average rating: {phone['rating']} out of 5. "
            f"Based on {phone['reviews']} customer reviews. "
            f"Average price: ${phone['price']}."
        )

        # Convert description to vector list
        embedding = get_embedding(description)

        # Pack into our batch collector array
        batch_data.append({
            "name": phone["name"],
            "embedding": json.dumps(embedding)
        })

       
        if len(batch_data) == batch_size or (i + 1) == total_phones:
            with driver.session() as session:
                session.run("""
                    UNWIND $batch AS item
                    MATCH (p:Phone {name: item.name})
                    SET p.embedding = item.embedding
                """, batch=batch_data)
            
            print(f"  Processed {i + 1}/{total_phones} phones")
            batch_data = []  # Reset the collector list

    print(f" All {total_phones} phones now have embeddings!")


# FUNCTION 4: Store review embeddings 
def store_review_embeddings(driver, limit=500):
    print("\n Creating embeddings for reviews...")

    with driver.session() as session:
        result = session.run("""
            MATCH (r:Review)
            RETURN r.id     AS id,
                   r.text   AS text,
                   r.rating AS rating
            LIMIT $limit
        """, limit=limit)
        reviews = [dict(record) for record in result]

    total_reviews = len(reviews)
    print(f"   Found {total_reviews} reviews to embed...")

    batch_data = []
    batch_size = 100

    for i, review in enumerate(reviews):
        embedding = get_embedding(review["text"])

        # Pack into batch array
        batch_data.append({
            "id": review["id"],
            "embedding": json.dumps(embedding)
        })

        # Commit batch when ready
        if len(batch_data) == batch_size or (i + 1) == total_reviews:
            with driver.session() as session:
                session.run("""
                    UNWIND $batch AS item
                    MATCH (r:Review {id: item.id})
                    SET r.embedding = item.embedding
                """, batch=batch_data)
            
            print(f"  Processed {i + 1}/{total_reviews} reviews")
            batch_data = []

    print(f" {total_reviews} reviews now have embeddings!")


# FUNCTION 5: Semantic search on phones
def semantic_search_phones(driver, question, top_k=5):
    # Step 1: Convert question to vector
    question_vector = get_embedding(question)

    # Step 2: Fetch all phone embeddings from Neo4j
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Phone)
            WHERE p.embedding IS NOT NULL
            RETURN p.name         AS name,
                   p.brand       AS brand,
                   p.avg_rating  AS rating,
                   p.review_count AS reviews,
                   p.avg_price   AS price,
                   p.embedding   AS embedding
        """)
        phones = [dict(record) for record in result]

    if not phones:
        return []

    # Step 3: Calculate cosine similarity for each phone
    scored_phones = []

    for phone in phones:
        # Parse embedding from JSON string back to list
        phone_vector = json.loads(phone["embedding"])

        # Calculate similarity score
        score = cosine_similarity(question_vector, phone_vector)

        scored_phones.append({
            "name"      : phone["name"],
            "brand"     : phone["brand"],
            "rating"    : phone["rating"],
            "reviews"   : phone["reviews"],
            "price"     : phone["price"],
            "similarity": round(score, 4),
        })

    # Step 4: Sort by similarity score (highest first)
    scored_phones.sort(key=lambda x: x["similarity"], reverse=True)

    # Return top K results
    return scored_phones[:top_k]


# FUNCTION 6: Semantic search on reviews 
def semantic_search_reviews(driver, question, top_k=5):
    question_vector = get_embedding(question)

    with driver.session() as session:
        result = session.run("""
            MATCH (p:Phone)-[:HAS_REVIEW]->(r:Review)
            WHERE r.embedding IS NOT NULL
            RETURN p.name   AS phone,
                   r.text   AS review,
                   r.rating AS rating,
                   r.embedding AS embedding
        """)
        reviews = [dict(record) for record in result]

    if not reviews:
        return []

    scored_reviews = []

    for rev in reviews:
        review_vector = json.loads(rev["embedding"])
        score         = cosine_similarity(question_vector, review_vector)

        scored_reviews.append({
            "phone"     : rev["phone"],
            "review"    : rev["review"],
            "rating"    : rev["rating"],
            "similarity": round(score, 4),
        })

    scored_reviews.sort(key=lambda x: x["similarity"], reverse=True)

    return scored_reviews[:top_k]


# FUNCTION 7: Build semantic context for RAG
def build_semantic_context(driver, question):
    context_parts = []

    # Semantic search on phones
    phones = semantic_search_phones(driver, question, top_k=5)

    if phones:
        context_parts.append("=== MOST RELEVANT PHONES (Semantic Search) ===")
        for p in phones:
            context_parts.append(
                f"• {p['name']} ({p['brand']}) | "
                f"Rating: {p['rating']}/5 | "
                f"Reviews: {p['reviews']} | "
                f"Price: ${p['price']} | "
                f"Relevance: {p['similarity']}"
            )

    # Semantic search on reviews
    reviews = semantic_search_reviews(driver, question, top_k=5)

    if reviews:
        context_parts.append("\n=== MOST RELEVANT REVIEWS (Semantic Search) ===")
        for r in reviews:
            context_parts.append(
                f"• {r['phone']} [{r['rating']}★] "
                f"(relevance: {r['similarity']}): {r['review']}"
            )

    return "\n".join(context_parts)


# TEST — python embeddings.py

if __name__ == "__main__":

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD"))
    )

    # Step 1: Store embeddings (run once)
    store_phone_embeddings(driver)
    store_review_embeddings(driver, limit=200)

    # Step 2: Test semantic search
    print("\n🔍 Testing semantic search...")
    print("\nQuery: 'best camera phone'")
    results = semantic_search_phones(driver, "best camera phone", top_k=3)
    for r in results:
        print(f"  {r['name']} — similarity: {r['similarity']}")

    print("\nQuery: 'phone with heating problems'")
    results = semantic_search_reviews(driver, "phone with heating problems", top_k=3)
    for r in results:
        print(f"  {r['phone']} [{r['rating']}★] — {r['review'][:80]}...")

    driver.close()