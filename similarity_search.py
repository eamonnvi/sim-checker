import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# Main Streamlit app
def main():
    st.title("Similarity Search")

    # Sidebar for file upload
    st.sidebar.header("Upload Embeddings and Metadata")
    metadata_file = st.sidebar.file_uploader("Upload Metadata CSV", type="csv")
    embeddings_file = st.sidebar.file_uploader("Upload Embeddings NPY", type="npy")

    # Input section
    st.header("Query Similarity")
    query_text = st.text_area("Enter the text to find similar highlights:")

    top_n = st.slider("Number of results to display", 1, 50, 10)

    if st.button("Search"):
        if not metadata_file or not embeddings_file:
            st.warning("Please upload both the metadata and embeddings files.")
            return

        if not query_text.strip():
            st.warning("Please enter some text to search.")
            return

        # Load metadata and embeddings
        try:
            highlights_df = pd.read_csv(metadata_file)
            embeddings = np.load(embeddings_file)
            st.write(f"Debug: Loaded {len(highlights_df)} entries from metadata.")
            st.write(f"Debug: Embeddings shape: {embeddings.shape}")
        except Exception as e:
            st.error(f"Failed to load files: {e}")
            return

        # Generate embedding for the query text
        query_embedding = generate_query_embedding(query_text)
        if query_embedding is None:
            st.error("Failed to generate embedding for the query.")
            return

        # Perform similarity search
        st.write("Calculating similarities...")
        similarities = cosine_similarity([query_embedding], embeddings).flatten()
        highlights_df["similarity"] = similarities

        # Sort and display top-N results
        top_results = highlights_df.nlargest(top_n, "similarity")
        for idx, row in top_results.iterrows():
            st.write(f"**{row['file']}** (Score: {row['similarity']:.4f})")
            with st.expander("View full text"):
                st.write(row["text"])

# Helper function to generate query embedding
def generate_query_embedding(query_text, model="text-embedding-ada-002"):
    from openai import OpenAI

    # Set your OpenAI API key
    client = OpenAI(api_key="REMOVED_KEYQMlXBLnf7EPnJUCHAJhnTXC22-gsZMFG6wediH1ShUNYSyjynETd3T3BlbkFJ2ktRAnE3QMtvhpIuQ5GOpYHPKA2R3Mxnrf-Q5MNCaqd35rK0lifM5xUiJx4_fMaxYM51n2ZusA")

    # Directory to save query embeddings
    QUERY_DIR = "/Users/eamonnvincent/Projects/Streamlit-Similarity/Queries"
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