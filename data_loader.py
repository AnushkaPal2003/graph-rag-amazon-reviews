# load and clean file
import pandas as pd
def load_data(filepath="data/Amazon_Unlocked_Mobile.csv"):
   
    print("Loading Dataset...")

    df = pd.read_csv(filepath)

    print(f" Total rows found: {len(df)}")

 
    imp_columns = [
        "Product Name",   
        "Brand Name",     
        "Price",          
        "Rating",         
        "Reviews",        
    ]

    df = df[imp_columns]

    # Column names 
    # "Product Name" → "product_name" 
    df.columns = ["product_name", "brand", "price", "rating", "review"]

    df = df.dropna(subset=["product_name", "brand", "rating", "review"])


    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["price"]  = pd.to_numeric(df["price"],  errors="coerce")

    
    df = df[df["rating"].between(1, 5)] 

    df["review"] = df["review"].astype(str).str.strip()

   
    df = df[df["review"] != ""]

  
    for col in ["product_name", "brand"]:
        df[col] = df[col].astype(str).str.strip()


    MAX_ROWS = 2000

    if len(df) > MAX_ROWS:
        
        df = df.sample(MAX_ROWS, random_state=42)


    df = df.reset_index(drop=True) # imp in Neo4j for ids

    print(f" Clean rows left: {len(df)}")
    print(f" Unique phones  : {df['product_name'].nunique()}")
    print(f"  Unique brands  : {df['brand'].nunique()}")

    return df


# FUNCTION 2: Find averaging rating of each phone

def compute_phone_stats(df):

    stats = df.groupby("product_name").agg(
        avg_rating   = ("rating", "mean"),   
        review_count = ("rating", "count"),  
        brand        = ("brand",  "first"),  
        avg_price    = ("price",  "mean"),    
    ).reset_index()

  
    stats["avg_rating"] = stats["avg_rating"].round(2)
    stats["avg_price"]  = stats["avg_price"].round(2)

    print("\n Top 10 Phones (by review count):")
    print(
        stats.sort_values("review_count", ascending=False)
             .head(10)
             .to_string(index=False) # Don't show row numbers
    )

    return stats


# FUNCTION 3: Find average rating for each brand

def compute_brand_stats(df):
    """
    Har brand ke liye average rating aur total reviews
    """

    stats = df.groupby("brand").agg(
        avg_rating   = ("rating", "mean"),
        review_count = ("rating", "count"),
        avg_price    = ("price",  "mean"),
    ).reset_index()

    stats["avg_rating"] = stats["avg_rating"].round(2)
    stats["avg_price"]  = stats["avg_price"].round(2)

    print("\n  Top 10 Brands:")
    print(
        stats.sort_values("review_count", ascending=False)
             .head(10)
             .to_string(index=False)
    )

    return stats


# DIRECTLY RUN KARO TEST KE LIYE: python data_loader.py

if __name__ == "__main__":

    print("=" * 55)
    print("  DATA LOADER TEST")
    print("=" * 55)

    df          = load_data()
    phone_stats = compute_phone_stats(df)
    brand_stats = compute_brand_stats(df)

    print("\n Data is loaded!")
   
