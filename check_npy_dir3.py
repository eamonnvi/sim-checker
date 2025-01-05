import os
import sys
import numpy as np
import pandas as pd

def report_embeddings(directory):
    """
    Generate a report of .npy files and their associated .csv files in the specified directory.
    
    :param directory: Path to the directory containing the files.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return
    
    # Find all .npy and .csv files in the directory
    npy_files = [f for f in os.listdir(directory) if f.startswith('embeddings_') and f.endswith('.npy')]
    csv_files = [f for f in os.listdir(directory) if f.startswith('metadata_') and f.endswith('.csv')]

    if not npy_files:
        print(f"No .npy files found in {directory}.")
        return
    
    print(f"Found {len(npy_files)} .npy files in {directory}:\n")
    report = []

    for npy_file in npy_files:
        npy_path = os.path.join(directory, npy_file)
        try:
            # Load the .npy file and get its shape
            embeddings = np.load(npy_path)
            shape = embeddings.shape
        except Exception as e:
            print(f"Error loading {npy_file}: {e}")
            continue

        # Identify the corresponding .csv file based on the shared timestamp and model
        timestamp_model_part = npy_file.replace('embeddings_', '').replace('.npy', '')
        possible_csv = f"metadata_{timestamp_model_part}.csv"
        csv_path = os.path.join(directory, possible_csv)
        csv_rows = None

        if possible_csv in csv_files:
            try:
                # Load the .csv file and count its rows
                df = pd.read_csv(csv_path)
                csv_rows = len(df)
            except Exception as e:
                print(f"Error loading {possible_csv}: {e}")
        
        # Add details to the report
        report.append({
            "Embedding File": npy_file,
            "Shape": shape,
            "CSV File": possible_csv if possible_csv in csv_files else "Not Found",
            "CSV Rows": csv_rows if csv_rows is not None else "N/A"
        })
    
    # Print the report
    print("{:<60} {:<15} {:<60} {:<10}".format("Embedding File", "Shape", "CSV File", "CSV Rows"))
    print("-" * 150)
    for item in report:
        print("{:<60} {:<15} {:<60} {:<10}".format(
            item["Embedding File"], 
            str(item["Shape"]), 
            item["CSV File"], 
            item["CSV Rows"]
        ))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python report_embeddings.py <directory_path>")
        sys.exit(1)

    directory = sys.argv[1]
    report_embeddings(directory)
