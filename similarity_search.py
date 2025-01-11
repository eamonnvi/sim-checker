import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import tempfile
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

def download_file(url):
    """Download a file and return the path to a temporary file."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(response.content)
            return temp_file.name
    except Exception as e:
        print(f"Error downloading the file: {e}")
        return None
        
# Fetch the pre-signed URLs instead of local paths
csv_url = "https://ev-sim-checker-for-streamlit.s3.eu-west-2.amazonaws.com/metadata_text-embedding-3-small_20250106_183211.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAU4ZT6XL2QBMUCW3Q%2F20250111%2Feu-west-2%2Fs3%2Faws4_request&X-Amz-Date=20250111T090329Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=f94df2d9f36652fbb11d57707ce24e1f790f943158040b1852f4803fecf57fcf"
npy_url = "https://ev-sim-checker-for-streamlit.s3.eu-west-2.amazonaws.com/embeddings_text-embedding-3-small_20250106_183211.npy?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAU4ZT6XL2QBMUCW3Q%2F20250111%2Feu-west-2%2Fs3%2Faws4_request&X-Amz-Date=20250111T093822Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=0fe7e73361b98f4f756d04669dc21cfc66a725b73cf48d08ac0c3ad5e0aaf51f"

# Download from S3 and load
csv_path = download_file(csv_url)
npy_path = download_file(npy_url)

# Load the files only if the download succeeds
if csv_path and npy_path:
    metadata_df = pd.read_csv(csv_path)
    embeddings = np.load(npy_path)
else:
    st.error("Failed to load one or both files.")

# Main Streamlit app
def main():
    st.title("Similarity Search")

    # Input section
    st.header("Query Similarity")
    query_text = st.text_area("Enter the text to find similar highlights:")

    top_n = st.slider("Number of results to display", 1, 50, 10)

    if st.button("Search"):
        if not query_text.strip():
            st.warning("Please enter some text to search.")
            return

        # Generate embedding for the query text
        query_embedding = generate_query_embedding(query_text)
        if query_embedding is None:
            st.error("Failed to generate embedding for the query.")
            return

        # Perform similarity search
        st.write("Calculating similarities...")
        similarities = cosine_similarity([query_embedding], embeddings).flatten()
        metadata_df["similarity"] = similarities

        # Sort and display top-N results
        top_results = metadata_df.nlargest(top_n, "similarity")
        for idx, row in top_results.iterrows():
            st.write(f"**{row['file']}** (Score: {row['similarity']:.4f})")
            with st.expander("View full text"):
                st.write(row["text"])

# Helper function to generate query embedding
def generate_query_embedding(query_text, model="text-embedding-3-small"):
    from openai import OpenAI

    # Set your OpenAI API key
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # Directory to save query embeddings
    QUERY_DIR = "./Queries"
    os.makedirs(QUERY_DIR, exist_ok=True)  # Ensure the directory exists

    try:
        # Generate embedding
        response = client.embeddings.create(model=model, input=query_text)
        embedding = np.array(response.data[0].embedding)

        # Save embedding with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_filename = os.path.join(QUERY_DIR, f"query_embedding_{timestamp}.npy")
        np.save(query_filename, embedding)

        # Save query metadata
        metadata_filename = os.path.join(QUERY_DIR, f"query_metadata_{timestamp}.csv")
        metadata = pd.DataFrame([{"text": query_text}])
        metadata.to_csv(metadata_filename, index=False)

        st.success(f"Query embedding saved: {query_filename}")
        st.success(f"Query metadata saved: {metadata_filename}")

        return embedding
    except Exception as e:
        st.error(f"Error generating embedding for query text: {e}")
        return None

if __name__ == "__main__":
    main()