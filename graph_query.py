

from neo4j import GraphDatabase


# FUNCTION 1: Get top rated phones

def get_top_phones(driver, limit=5):
    """
    Returns top N phones sorted by average rating.
    """

    query = """
    MATCH (p:Phone)
    RETURN p.name         AS phone,
           p.brand        AS brand,
           p.avg_rating   AS avg_rating,
           p.review_count AS reviews,
           p.avg_price    AS price
    ORDER BY p.avg_rating DESC
    LIMIT $limit
    """

    with driver.session() as session:
        result = session.run(query, limit=limit)
        return [dict(record) for record in result]



# FUNCTION 2: Get all phones by a brand

def get_phones_by_brand(driver, brand_name):
    """
    Returns all phones made by a specific brand.
    Graph traversal: (Brand)-[:MAKES]->(Phone)
    """

    query = """
    MATCH (b:Brand)-[:MAKES]->(p:Phone)
    WHERE toLower(b.name) CONTAINS toLower($brand)
    RETURN b.name        AS brand,
           p.name        AS phone,
           p.avg_rating  AS avg_rating,
           p.avg_price   AS price,
           p.review_count AS reviews
    ORDER BY p.avg_rating DESC
    """

    with driver.session() as session:
        result = session.run(query, brand=brand_name)
        return [dict(record) for record in result]


# FUNCTION 3: Get reviews for a specific phone

def get_phone_reviews(driver, search_keyword, limit=5):
    """
    Returns reviews for phones matching the search keyword.
    Graph traversal: (Phone)-[:HAS_REVIEW]->(Review)
    """

    query = """
    MATCH (p:Phone)-[:HAS_REVIEW]->(r:Review)
    WHERE toLower(p.name) CONTAINS toLower($search) OR toLower(p.brand) CONTAINS toLower($search)
    RETURN p.name   AS phone,
           r.text   AS review,
           r.rating AS rating
    ORDER BY r.rating DESC
    LIMIT $limit
    """

    with driver.session() as session:
        result = session.run(query, search=search_keyword, limit=limit)
        return [dict(record) for record in result]


# FUNCTION 4: Compare all brands

def get_brand_comparison(driver):
    """
    Returns a comparison table of all brands —
    how many phones, average rating, total reviews.
    """

    query = """
    MATCH (b:Brand)-[:MAKES]->(p:Phone)
    RETURN b.name                      AS brand,
           count(p)                    AS phone_count,
           round(avg(p.avg_rating), 2) AS avg_rating,
           sum(p.review_count)         AS total_reviews
    ORDER BY avg_rating DESC
    LIMIT 20
    """

    with driver.session() as session:
        result = session.run(query)
        return [dict(record) for record in result]


# FUNCTION 5: Smart context builder — HEART of Graph RAG


def build_context_for_question(driver, question, query_type="GENERAL"):
    """
    Looks at the router's query_type and executes ONLY the 
    specific Cypher query needed. This prevents database bottlenecks.
    """
    question_lower = question.lower()
    context_parts  = []

    # Rule 1 — User is asking about best / top phones
    if query_type == "TOP_PHONES":
        phones = get_top_phones(driver, limit=5)
        context_parts.append("=== TOP RATED PHONES ===")
        for p in phones:
            context_parts.append(
                f"• {p['phone']} ({p['brand']}) | Rating: {p['avg_rating']}/5 "
                f"| Reviews: {p['reviews']} | Price: ${p['price']}"
            )

    # Rule 2 — User is asking about a specific brand
    elif query_type == "BRAND_QUERY":
        for brand in ["apple", "samsung", "google", "oneplus", "motorola", "xiaomi"]:
            if brand in question_lower:
                phones = get_phones_by_brand(driver, brand)
                if phones:
                    context_parts.append(f"\n=== {brand.upper()} PHONES ===")
                    for p in phones[:10]:  # Limit output count to prevent context blowup
                        context_parts.append(
                            f"• {p['phone']} | Rating: {p['avg_rating']}/5 | Price: ${p['price']}"
                        )
                break

    # Rule 3 — User is asking about reviews / consumer feedback
    elif query_type == "PHONE_REVIEWS":
        search_target = None
        # Identify target model or brand within text string
        for keyword in ["apple", "samsung", "google", "oneplus", "motorola", "xiaomi", "iphone", "galaxy", "pixel"]:
            if keyword in question_lower:
                search_target = keyword
                break
        
        if not search_target:
            search_target = question_lower  

        reviews = get_phone_reviews(driver, search_target, limit=5)
        if reviews:
            context_parts.append(f"\n=== {search_target.upper()} REVIEWS ===")
            for r in reviews:
                context_parts.append(f"• {r['phone']} [{r['rating']}★]: {r['review']}")

    # Rule 4 — User wants to compare brands
    elif query_type == "BRAND_COMPARISON":
        brands = get_brand_comparison(driver)
        context_parts.append("\n=== BRAND COMPARISON ===")
        for b in brands:
            context_parts.append(
                f"• {b['brand']} | Phones: {b['phone_count']} "
                f"| Avg Rating: {b['avg_rating']}/5 | Total Reviews: {b['total_reviews']}"
            )

   
    else:
        phones = get_top_phones(driver, limit=3)
        context_parts.append("=== GENERAL DATA ===")
        for p in phones:
            context_parts.append(f"• {p['phone']} — {p['avg_rating']}/5 stars")

    return "\n".join(context_parts)


# TEST — python graph_query.py

if __name__ == "__main__":

    import os
    from dotenv import load_dotenv
    load_dotenv()

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD"))
    )

    print("=== Top 3 Phones ===")
    for p in get_top_phones(driver, 3):
        print(p)

    print("\n=== Samsung Phones ===")
    for p in get_phones_by_brand(driver, "samsung"):
        print(p)

    print("\n=== Context for: best samsung phone ===")
    print(build_context_for_question(driver, "best samsung phone", query_type="TOP_PHONES"))

    driver.close()