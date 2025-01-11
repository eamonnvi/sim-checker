import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import tempfile
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# Pre-signed URLs for S3 files
csv_url = "https://ev-sim-checker-for-streamlit.s3.eu-west-2.amazonaws.com/metadata_text-embedding-3-small_20250106_183211.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAU4ZT6XL2QBMUCW3Q%2F20250111%2Feu-west-2%2Fs3%2Faws4_request&X-Amz-Date=20250111T090329Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=f94df2d9f36652fbb11d57707ce24e1f790f943158040b1852f4803fecf57fcf"
npy_url = "https://ev-sim-checker-for-streamlit.s3.eu-west-2.amazonaws.com/embeddings_text-embedding-3-small_20250106_183211.npy?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAU4ZT6XL2QBMUCW3Q%2F20250111%2Feu-west-2%2Fs3%2Faws4_request&X-Amz-Date=20250111T093822Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=0fe7e73361b98f4f756d04669dc21cfc66a725b73cf48d08ac0c3ad5e0aaf51f"

# Directory for storing novel query embeddings locally
QUERY_DIR = "./Queries"
os.makedirs(QUERY_DIR, exist_ok=True)  # Ensure the directory exists

def download_file(url):
    """Download a file from a URL and return the path to a temporary file."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(response.content)
            return temp_file.name
    except Exception as e:
        st.error(f"Error downloading the file: {e}")
        return None

def load_data():
    """Lazy load the CSV and NPY files only when needed."""
    st.write("Downloading data from S3...")
    csv_path = download_file(csv_url)
    npy_path = download_file(npy_url)
    if csv_path and npy_path:
        metadata_df = pd.read_csv(csv_path)
        embeddings = np.load(npy_path)
        return metadata_df, embeddings
    else:
        st.error("Failed to load one or both files.")
        return None, None

def generate_query_embedding(query_text, model="text-embedding-3-small"):
    """Generate and save the query embedding with a timestamp."""
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    try:
        response = client.embeddings.create(model=model, input=query_text)
        embedding = np.array(response.data[0].embedding)

        # Save query embedding and metadata with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_filename = os.path.join(QUERY_DIR, f"query_embedding_{timestamp}.npy")
        np.save(query_filename, embedding)

        metadata_filename = os.path.join(QUERY_DIR, f"query_metadata_{timestamp}.csv")
        metadata = pd.DataFrame([{"text": query_text}])
        metadata.to_csv(metadata_filename, index=False)

        st.success(f"Query embedding saved: {query_filename}")
        st.success(f"Query metadata saved: {metadata_filename}")

        return embedding
    except Exception as e:
        st.error(f"Error generating embedding for query text: {e}")
        return None

# Main Streamlit App
def main():
    st.title("Similarity Search with Lazy Loading and Query Saving")

    query_text = st.text_area("Enter the text to search:")

    if st.button("Search"):
        metadata_df, embeddings = load_data()
        if metadata_df is None:
            return
        
        # Generate query embedding and save it
        query_embedding = generate_query_embedding(query_text)
        if query_embedding is None:
            return

        # Perform similarity search
        similarities = cosine_similarity([query_embedding], embeddings).flatten()
        metadata_df["similarity"] = similarities

        # Display results
        top_results = metadata_df.nlargest(10, "similarity")
        for idx, row in top_results.iterrows():
            st.write(f"**{row['file']}** (Score: {row['similarity']:.4f})")
            with st.expander("View full text"):
                st.write(row["text"])

if __name__ == "__main__":
    main()