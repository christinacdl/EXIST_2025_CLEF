import pandas as pd
import chardet
import matplotlib.pyplot as plt
import glob
import os
import argparse

def detect_encoding(file_path):
    """Detects file encoding."""
    with open(file_path, 'rb') as f:
        rawdata = f.read(10000)  # Read a portion of the file
    return chardet.detect(rawdata)['encoding']

def read_csv_with_encoding(file_path):
    """Reads CSV file with detected encoding."""
    encoding = detect_encoding(file_path)
    return pd.read_csv(file_path, encoding=encoding)

def merge_csv_files(files, merge_on='id'):
    """Merges multiple CSV files on a given column."""
    df_list = [read_csv_with_encoding(file) for file in files]
    merged_df = df_list[0]
    for df in df_list[1:]:
        merged_df = pd.merge(merged_df, df, on=merge_on, how='outer')
    return merged_df

def plot_label_distribution(df, label_column, output_path):
    """Plots label distribution and saves the figure."""
    if label_column not in df.columns:
        print(f"Column '{label_column}' not found in the merged CSV. Skipping plot.")
        return
    
    plt.figure(figsize=(8, 5))
    df[label_column].value_counts().plot(kind='bar')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title('Label Distribution')
    plt.xticks(rotation=45)
    plt.savefig(output_path)
    plt.close()
    print(f"Label distribution plot saved as {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Merge CSV files and analyze label distribution.")
    parser.add_argument('--input_folder', type=str, required=True, help="Folder containing CSV files to merge.")
    parser.add_argument('--output_file', type=str, default='merged_output.csv', help="Output file name for merged CSV.")
    parser.add_argument('--label_column', type=str, default='label', help="Column name for label distribution analysis.")
    parser.add_argument('--plot_file', type=str, default='label_distribution.png', help="Output file for label distribution plot.")
    parser.add_argument('--merge_on', type=str, default='id', help="Column name to merge CSV files on.")

    args = parser.parse_args()

    # Get list of CSV files
    files = glob.glob(os.path.join(args.input_folder, '*.csv'))
    
    if not files:
        print("No CSV files found in the folder.")
        return
    
    merged_df = merge_csv_files(files, args.merge_on)
    merged_df.to_csv(args.output_file, index=False)
    print(f"Merged CSV saved as {args.output_file}")

    # Generate label distribution plot
    if args.label_column in merged_df.columns:
        plot_label_distribution(merged_df, args.label_column, args.plot_file)

if __name__ == "__main__":
    main()

# python merge_csv.py --input_folder path/to/csvs --output_file merged.csv --label_column label --plot_file labels.png --merge_on id
