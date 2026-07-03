import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import ast

def split_dataset(csv_path: str, output_dir: str, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    print("--- Starting Dataset Splitting (Patient Group Split) ---")
    
    # Load dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows.")
    
    # 1. Clean / verify file paths
    # We already know all df['filename'] exist in 'preprocessed_images'
    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Get unique patients and their main labels for stratification if possible,
    # or perform a patient-grouped random split.
    # Group by patient ID to ensure no data leakage.
    unique_ids = df['ID'].unique()
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(unique_ids)
    
    n_patients = len(unique_ids)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    train_ids = unique_ids[:n_train]
    val_ids = unique_ids[n_train:n_train + n_val]
    test_ids = unique_ids[n_train + n_val:]
    
    # Map back to dataframe
    train_df = df[df['ID'].isin(train_ids)].copy()
    val_df = df[df['ID'].isin(val_ids)].copy()
    test_df = df[df['ID'].isin(test_ids)].copy()
    
    print(f"Split results:")
    print(f"  Train patients: {len(train_ids)} (rows: {len(train_df)})")
    print(f"  Val patients: {len(val_ids)} (rows: {len(val_df)})")
    print(f"  Test patients: {len(test_ids)} (rows: {len(test_df)})")
    
    # 3. Verify class distribution across splits
    labels = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    print("\nClass distribution in Train set:")
    train_counts = train_df[labels].sum()
    for l in labels:
        print(f"  {l}: {train_counts[l]} ({train_counts[l]/len(train_df)*100:.2f}%)")
        
    print("\nClass distribution in Val set:")
    val_counts = val_df[labels].sum()
    for l in labels:
        print(f"  {l}: {val_counts[l]} ({val_counts[l]/len(val_df)*100:.2f}%)")
        
    print("\nClass distribution in Test set:")
    test_counts = test_df[labels].sum()
    for l in labels:
        print(f"  {l}: {test_counts[l]} ({test_counts[l]/len(test_df)*100:.2f}%)")
        
    # 4. Save splits
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSaved train, val, and test splits to {output_dir}")
    print("--- Dataset Splitting Completed ---")

if __name__ == "__main__":
    csv_path = "full_df.csv"
    output_dir = "data"
    split_dataset(csv_path, output_dir)
