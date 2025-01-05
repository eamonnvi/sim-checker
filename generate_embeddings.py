import os
import sys
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from openai import OpenAI

# Set your OpenAI API key
client = OpenAI(api_key="REMOVED_KEYQMlXBLnf7EPnJUCHAJhnTXC22-gsZMFG6wediH1ShUNYSyjynETd3T3BlbkFJ2ktRAnE3QMtvhpIuQ5GOpYHPKA2R3Mxnrf-Q5MNCaqd35rK0lifM5xUiJx4_fMaxYM51n2ZusA")

def load_highlights(directory):
    """
    Load markdown files into a DataFrame.
    :param directory: Path to directory containing markdown files.
    :return: DataFrame with 'file' and 'text' columns.
    """
    markdown_files = glob.glob(f"{directory}/*.md")
    if not markdown_files:
        print(f"No markdown files found in {directory}.")
        sys.exit(1)
    
    data = []
    for file_path in markdown_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                data.append({"file": os.path.basename(file_path), "text": text})
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return pd.DataFrame(data)

def generate_embeddings(df, model="text-embedding-ada-002"):
    """
    Generate embeddings for a DataFrame of text using OpenAI API.
    :param df: DataFrame with 'text' column.
    :param model: Embedding model to use.
    :return: Numpy array of embeddings.
    """
    embeddings = []
    for idx, row in df.iterrows():
        try:
            response = client.embeddings.create(model=model, input=row["text"])
            embedding = np.array(response.data[0].embedding)
            embeddings.append(embedding)
            print(f"Processed {row['file']} ({idx + 1}/{len(df)})")
        except Exception as e:
            print(f"Error processing {row['file']}: {e}")
    return np.array(embeddings)

def main(directory_path, embedding_model):
    # Load markdown files into a DataFrame
    highlights_df = load_highlights(directory_path)
    print(f"Loaded {len(highlights_df)} markdown files.")

    # Generate embeddings
    embeddings = generate_embeddings(highlights_df, model=embedding_model)

    # Save embeddings and metadata with timestamped filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "Embeddings"
    os.makedirs(output_dir, exist_ok=True)
    npy_filename = os.path.join(output_dir, f"embeddings_{embedding_model}_{timestamp}.npy")
    csv_filename = os.path.join(output_dir, f"metadata_{embedding_model}_{timestamp}.csv")

    np.save(npy_filename, embeddings)
    highlights_df.to_csv(csv_filename, index=False)

    print(f"Embeddings and metadata saved to {npy_filename} and {csv_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_embeddings.py <directory_path> [embedding_model]")
        sys.exit(1)

    directory_path = sys.argv[1]
    embedding_model = sys.argv[2] if len(sys.argv) > 2 else "text-embedding-ada-002"
    main(directory_path, embedding_model)
