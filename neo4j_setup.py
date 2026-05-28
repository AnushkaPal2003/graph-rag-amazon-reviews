
import os
from neo4j import GraphDatabase   
from dotenv import load_dotenv     
from data_loader import (
    load_data,
    compute_phone_stats,
    compute_brand_stats,
)

load_dotenv()


# FUNCTION 1: Connect to Neo4j

def connect_to_neo4j():
   
    uri      = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    if not uri or not password:
        print(" No Neo4j details in .env!")
        return None

    try:
        
        driver = GraphDatabase.driver(uri, auth=(username, password))

       
        driver.verify_connectivity()

        print(" Connected to Neo4j")
        return driver

    except Exception as e:
        print(f" Connection failed!!: {e}")
        return None



# FUNCTION 2: Clean the database

def clear_database(driver):
    
    with driver.session() as session: # context Manager of Python
        session.run("MATCH (n) DETACH DELETE n") # Detach beacuse first relationships are deleted and then Nodes

    print("Old data has been deleted!!")



# FUNCTION 3: Create Brand Nodes
def create_brand_nodes(driver, df):
   

    brands = df["brand"].unique()

    with driver.session() as session:
        for brand in brands:
           session.run(
                "MERGE (:Brand {name: $name})",
                name=brand
            )

    print(f" {len(brands)} Brands created!!")


# FUNCTION 4: Create Phone nodes 

def create_phone_nodes(driver, phone_stats):

    with driver.session() as session:
        for _, row in phone_stats.iterrows():

            avg_price = float(row["avg_price"]) if str(row["avg_price"]) != "nan" else 0.0

            session.run(
                """
                MERGE (p:Phone {name: $name})
                SET p.avg_rating   = $avg_rating,
                    p.review_count = $review_count,
                    p.avg_price    = $avg_price,
                    p.brand        = $brand
                """,
                name         = row["product_name"],
                avg_rating   = float(row["avg_rating"]),
                review_count = int(row["review_count"]),
                avg_price    = avg_price,
                brand        = row["brand"],
            )

    print(f" {len(phone_stats)} Phone nodes have been created!!")



# FUNCTION 5: Create Review nodes 

def create_review_nodes(driver, df):
  
    with driver.session() as session:
        for i, row in df.iterrows():
            session.run(
                """
                CREATE (r:Review {
                    id     : $id,
                    text   : $text,
                    rating : $rating
                })
                """,
                id     = int(i),
                text   = str(row["review"])[:500], 
                rating = int(row["rating"]),
            )

    print(f" {len(df)} Review nodes are created!!")



# FUNCTION 6: Relationships between nodes and relationships
def create_relationships(driver, df):

    with driver.session() as session:

        # Relationship 1: Brand MAKES Phone
        brand_phone_pairs = df[["brand", "product_name"]].drop_duplicates()

        for _, row in brand_phone_pairs.iterrows():
            session.run(
                """
                MATCH (b:Brand {name: $brand})
                MATCH (p:Phone {name: $phone})
                MERGE (b)-[:MAKES]->(p)
                """,
                brand = row["brand"],
                phone = row["product_name"],
            )

        # Relationship 2: Phone HAS_REVIEW Review
        for i, row in df.iterrows():
            session.run(
                """
                MATCH (p:Phone {name: $phone})
                MATCH (r:Review {id: $id})
                MERGE (p)-[:HAS_REVIEW]->(r)
                """,
                phone = row["product_name"],
                id    = int(i),
            )

    print(" All Relationships are created")


# FUNCTION 7: check Graph  

def verify_graph(driver):

    with driver.session() as session:

        # Nodes count
        result = session.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count"
        )
        print("\n Graph Summary:")
        for record in result:
            print(f"   {record['label']}: {record['count']} nodes")

        # Relationships count
        result = session.run(
            "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS count"
        )
        print("\n Relationships:")
        for record in result:
            print(f"   {record['rel']}: {record['count']}")


if __name__ == "__main__":

    print("=" * 55)
    print("  KNOWLEDGE GRAPH IS BUILDING!!")
    print("=" * 55)

   
    df          = load_data()
    phone_stats = compute_phone_stats(df)
    brand_stats = compute_brand_stats(df)

 
    driver = connect_to_neo4j()
    if not driver:
        exit(1)


    clear_database(driver)
    create_brand_nodes(driver, df)
    create_phone_nodes(driver, phone_stats)
    create_review_nodes(driver, df)
    create_relationships(driver, df)
    verify_graph(driver)

    driver.close()
    print("\n Graph Created!!")
