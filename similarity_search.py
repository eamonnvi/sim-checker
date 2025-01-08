import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# Function to download files from Dropbox
def download_file(url, file_path):
    """Download a file from a URL if it doesn't already exist."""
    if not os.path.exists(file_path):
        st.info(f"Downloading file from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            f.write(response.content)
        st.success(f"File downloaded and saved: {file_path}")

# Dropbox direct links (replace with your actual links)
csv_url = "https://www.dropbox.com/scl/fi/m03h7ok57wiex236s4fa2/metadata_text-embedding-3-small_20250106_183211.csv?rlkey=kb2c9iqfbp0lfj0n26fv88c3y&st=tud882v9&dl=1"
npy_url = "https://www.dropbox.com/scl/fi/v8wr4ky071w8gu828ru6q/embeddings_text-embedding-3-small_20250106_183211.npy?rlkey=fzuq7s8l0jex4jje1byp9vn91&st=kadw8pkn&dl=1"

# Local file paths
csv_path = "Embed-Output/metadata_text-embedding-3-small_20250106_183211.csv"
npy_path = "Embed-Output/embeddings_text-embedding-3-small.npy"

# Ensure both files are downloaded before loading
download_file(csv_url, csv_path)
download_file(npy_url, npy_path)

# Load the files directly
try:
    metadata_df = pd.read_csv(csv_path)
    embeddings = np.load(npy_path)
    st.success(f"Loaded {len(metadata_df)} metadata entries and embeddings with shape {embeddings.shape}.")
except Exception as e:
    st.error(f"Error loading files: {e}")

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
    client = OpenAI(api_key="REMOVED_KEYQMlXBLnf7EPnJUCHAJhnTXC22-gsZMFG6wediH1ShUNYSyjynETd3T3BlbkFJ2ktRAnE3QMtvhpIuQ5GOpYHPKA2R3Mxnrf-Q5MNCaqd35rK0lifM5xUiJx4_fMaxYM51n2ZusA")

    # Directory to save query embeddings
    QUERY_DIR = "/Users/eamonnvincent/Dev/Queries"
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