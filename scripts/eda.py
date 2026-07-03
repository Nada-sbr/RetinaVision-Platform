import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(csv_path: str, output_dir: str):
    print("--- Starting Exploratory Data Analysis ---")
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the dataset
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded. Total rows: {len(df)}")
    
    # 1. Check basic info
    print("\n--- Basic Information ---")
    print(df.info())
    
    # Check for missing values
    missing = df.isnull().sum()
    print("\n--- Missing Values ---")
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")
    
    # 2. Demographic Analysis
    print("\n--- Demographics: Age ---")
    age_stats = df['Patient Age'].describe()
    print(age_stats)
    
    print("\n--- Demographics: Sex ---")
    sex_counts = df['Patient Sex'].value_counts()
    sex_pcts = df['Patient Sex'].value_counts(normalize=True) * 100
    for idx in sex_counts.index:
        print(f"{idx}: {sex_counts[idx]} ({sex_pcts[idx]:.2f}%)")
        
    # Plot Demographics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df['Patient Age'], bins=20, kde=True, color='skyblue')
    plt.title('Patient Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    sns.barplot(x=sex_counts.index, y=sex_counts.values, palette='pastel')
    plt.title('Patient Sex Distribution')
    plt.xlabel('Sex')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'demographics_distribution.png'))
    plt.close()
    print(f"Demographics plot saved to {os.path.join(output_dir, 'demographics_distribution.png')}")
    
    # 3. Label / Class Distribution
    labels = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    class_counts = df[labels].sum()
    class_pcts = (class_counts / len(df)) * 100
    
    print("\n--- Disease Class Distribution ---")
    for label in labels:
        print(f"Class {label}: {class_counts[label]} occurrences ({class_pcts[label]:.2f}%)")
        
    # Plot Class Distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
    plt.title('Disease Class Distribution (Multi-label)')
    plt.xlabel('Disease Label')
    plt.ylabel('Number of Patients')
    # Add values on top of bars
    for i, v in enumerate(class_counts.values):
        plt.text(i, v + 20, str(int(v)), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'disease_class_distribution.png'))
    plt.close()
    print(f"Disease distribution plot saved to {os.path.join(output_dir, 'disease_class_distribution.png')}")
    
    # 4. Multi-label Co-occurrence Analysis
    print("\n--- Multi-label Co-occurrence (Correlations) ---")
    co_occurrence = df[labels].T.dot(df[labels])
    print(co_occurrence)
    
    # Plot Co-occurrence Correlation Heatmap
    corr = df[labels].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Disease Labels Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'labels_correlation_heatmap.png'))
    plt.close()
    print(f"Correlation heatmap saved to {os.path.join(output_dir, 'labels_correlation_heatmap.png')}")
    
    # 5. Check if files exist
    print("\n--- File Verification ---")
    # Verify preprocessed_images existence
    preprocessed_dir = "preprocessed_images"
    left_exist = 0
    right_exist = 0
    total_rows = len(df)
    
    for idx, row in df.iterrows():
        left_img = os.path.join(preprocessed_dir, row['Left-Fundus'])
        right_img = os.path.join(preprocessed_dir, row['Right-Fundus'])
        if os.path.exists(left_img):
            left_exist += 1
        if os.path.exists(right_img):
            right_exist += 1
            
    print(f"Left images found in preprocessed_images: {left_exist}/{total_rows}")
    print(f"Right images found in preprocessed_images: {right_exist}/{total_rows}")
    
    print("\n--- Exploratory Data Analysis Completed ---")

if __name__ == "__main__":
    csv_path = "full_df.csv"
    output_dir = "plots"
    run_eda(csv_path, output_dir)
