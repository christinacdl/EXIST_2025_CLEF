import os
import re
import ast
import pandas as pd
import numpy as np
import argparse
import chardet
import emoji
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from textblob import TextBlob
from collections import Counter
import wordsegment as ws
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

LABEL_LIST = [
    "IDEOLOGICAL-INEQUALITY",
    "STEREOTYPING-DOMINANCE",
    "OBJECTIFICATION",
    "SEXUAL-VIOLENCE",
    "MISOGYNY-NON-SEXUAL-VIOLENCE",
    "NO"]


def create_hard_labels(df):
    """Convert `hard_label` to multi-hot encoding in a separate DataFrame."""
    df["hard_labels"] = df["hard_label"].fillna("[]").apply(
        lambda x: [1 if label in (ast.literal_eval(x) if isinstance(x, str) else x or []) else 0 for label in LABEL_LIST]
    )
    return df


def create_soft_labels(df):
    """Convert `soft_label` to probability vectors in a separate DataFrame."""
    df["soft_labels"] = df["soft_label"].fillna("{}").apply(
        lambda x: [ast.literal_eval(x).get(label, 0) if isinstance(x, str) else x.get(label, 0) if isinstance(x, dict) else 0 for label in LABEL_LIST]
    )
    return df


# PRE-PROCESSING
text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize = ['user', 'url', 'email'],

    # terms that will be annotated
    #annotate = {'hashtag'},  #{'allcaps', 'repeated', 'elongated'},

    # corpus from which the word statistics are going to be used for word segmentation
    segmenter = 'twitter',  # or 'english'

    # corpus from which the word statistics are going to be used for spell correction
    corrector = 'twitter',  # or 'english'

    fix_html = False,              # fix HTML tokens
    fix_text = False,              # fix text
    unpack_hashtags = True,       # perform word segmentation on hashtags
    unpack_contractions = False,  # Unpack contractions (can't -> can not)
    spell_correct_elong = False,   # spell correction for elongated words

    tokenizer = SocialTokenizer(lowercase = False).tokenize)


# def sep_digits(x):
#     return " ".join(re.split('(\d+)', x))


# def sep_punc(x):
#     punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~؛،؟؛.»«”'
#     out = []
#     for char in x:
#         if char in punc:
#             out.append(' ' + char + ' ')
#         else:
#             out.append(char)
#     return ''.join(out)


ws.load()
def segment_hashtags(text):
    text = re.sub(r'#\S+', lambda match: ' '.join(ws.segment(match.group())), text)
    return text


def emojis_into_text(sentence):
    demojized_sent = emoji.demojize(sentence)
    emoji_txt = re.sub(r':[\S]+:', lambda x: x.group().replace('_', ' ').replace('-', ' ').replace(':', ''), demojized_sent)
    return emoji_txt


def preprocessing(text):

    # Convert the emojis into their textual representation
    text = emojis_into_text(text)

    # # Replace '&amp;' with 'and'
    text = re.sub(r'&amp;','and', text)
    text = re.sub(r'&','and', text)

    # # # Replace the unicode apostrophe
    text = re.sub(r"’","'", text)
    text = re.sub(r'“','"', text)

    # Replace consecutive non-ASCII characters with whitespace
    text = re.sub(r'[^\x00-\x7F]+',' ', text)

    text = re.sub(' +',' ', text) 

    # Apply the text processor from ekphrasis library
    text = ' '.join(text_processor.pre_process_doc(text))

    # Apply hashtag segmentation
    text = segment_hashtags(text)

    return text


def detect_encoding(file_path, sample_size=10000):
    """Detect file encoding using chardet."""
    with open(file_path, 'rb') as f:
        rawdata = f.read(sample_size)
    result = chardet.detect(rawdata)
    return result['encoding']


def analyze_sentiment(text):
    """Determine sentiment of a given text using VADER (for English) and TextBlob (for others)."""
    if isinstance(text, str) and text.strip():
        try:
            lang = TextBlob(text).detect_language()
        except:
            lang = "en"  # Default to English if detection fails

        if lang == "en":
            analyzer = SentimentIntensityAnalyzer()
            score = analyzer.polarity_scores(text)['compound']
        else:
            score = TextBlob(text).sentiment.polarity  # Use TextBlob for non-English
        
        return score
    return 0  # Default for empty or NaN texts


def save_sentiment_distribution(df, output_path):
    """Generate and save sentiment distribution as an image."""
    plt.figure(figsize=(14, 6))
    sentiment_counts = df['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0).reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    ax = sns.barplot(x="Sentiment", y="Count", hue="Sentiment", data=sentiment_counts, palette={'Positive': 'green', 'Neutral': 'blue', 'Negative': 'red'}, legend=False)
    for bar, value in zip(ax.patches, sentiment_counts["Count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10, str(int(value)), ha="center", fontsize=12) #fontweight='bold'
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.ylim(0, max(sentiment_counts["Count"]) + 0.7)  
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_label_distribution(df, label_column, output_path, soft_labels=False):
    """Generate and save label distribution for multi-label classification.

    Args:
        df (pd.DataFrame): The dataset.
        label_column (str): The column containing the labels.
        output_path (str): Path to save the plot.
        soft_labels (bool): If True, handles soft labels (probability distributions).
    """
    label_counts = Counter()

    if soft_labels:
        # Soft labels: Sum the probabilities of each label across all rows
        for labels in df[label_column].dropna():
            if isinstance(labels, dict):  # Ensure labels are stored as dictionaries
                for label, value in labels.items():
                    label_counts[label] += value  # Sum probability values
    else:
        # Hard labels: Flatten multi-label lists into a single list
        for labels in df[label_column].dropna():
            if isinstance(labels, list):
                label_counts.update(labels)

    # Convert to DataFrame for visualization
    label_df = pd.DataFrame(label_counts.items(), columns=["Label", "Count"]).sort_values(by="Count", ascending=False)

    # Plot distribution with unique colors for each label
    plt.figure(figsize=(14, 7))
    palette = sns.color_palette("husl", len(label_df))  # Unique color for each label
    ax = sns.barplot(x="Count", y="Label", hue="Label", data=label_df, palette=palette, legend=False)

    for index, value in enumerate(label_df["Count"]):
        ax.text(value + 0.5, index, f"{value:.2f}" if soft_labels else f"{int(value)}", va="center", fontsize=12)

    plt.title(f"Label Distribution: {label_column}")
    plt.xlabel("Count" if not soft_labels else "Total Probability Sum")
    plt.ylabel("Labels")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def analyze_dataset(df, text_column, label_column, output_dir, dataset_name, soft_labels=False):
    """Perform dataset analysis: missing values, duplicates, and save label distribution."""
    print(f"\nAnalyzing {dataset_name} dataset...")

    print(f"\nOriginal Dataset: {df.shape[0]} rows.")

    # Check missing values and duplicates
    print("Missing Values:\n", df.isnull().sum(), "\n")
    print(f"Duplicate Rows: {df.duplicated(subset=[text_column]).sum()} \n")

    # Drop duplicates based on text column
    df = df.drop_duplicates(subset=[text_column])

    # Remove rows where text column is null or empty
    df = df[df[text_column].astype(str).str.strip() != ""]
    df = df.dropna(subset=[text_column, label_column])

    print(f"\nDataset cleaned: {df.shape[0]} rows remaining after duplicate & null removal.\n")

    # Text length statistics
    df['text_length'] = df[text_column].astype(str).apply(len)
    print("Text Length Statistics:\n", df['text_length'].describe(), "\n")

    print(f"\nPre-processing {dataset_name} dataset...")
    df[text_column] = df[text_column].apply(lambda x: preprocessing(x))

    # Check and save Label Distribution
    print(f"\nSaving Label Distribution for {dataset_name} dataset...")
    save_label_distribution(df, label_column, os.path.join(output_dir, f"{dataset_name}_label_distribution.png"), soft_labels)

    # Perform Sentiment Analysis
    print(f"\nPerforming Sentiment Analysis on {dataset_name} dataset...")
    df['sentiment_score'] = df[text_column].astype(str).apply(analyze_sentiment)
    df['sentiment'] = df['sentiment_score'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

    # Save Sentiment Distribution Plot
    print(f"\nSaving Sentiment Distribution for {dataset_name} dataset...")
    save_sentiment_distribution(df, os.path.join(output_dir, f"{dataset_name}_sentiment_distribution.png"))

    return df


def load_exist2025_dataset(train_dir, dev_dir, merge_data=False, language="all", analyze=True, text_column="tweet", label_column="hard_label"):
    """
    Load, process, and optionally analyze the EXIST 2024 dataset.

    Args:
        train_dir (str): Path to the training dataset directory.
        dev_dir (str): Path to the development dataset directory.
        merge_data (bool): Whether to merge training and development datasets.
        language (str): Choose "all", "en", or "es" (default: "all").
        analyze (bool): Perform dataset analysis before saving.
        text_column (str): Column name for the text data.
        label_column (str): Column name for the label data.

    Returns:
        Processed DataFrame(s).
    """
    train_dir = os.path.abspath(train_dir)
    dev_dir = os.path.abspath(dev_dir)

    # Load training and development data
    tr_df = pd.read_json(os.path.join(train_dir, 'EXIST2025_training.json')).T.reset_index(drop=True)
    dev_df = pd.read_json(os.path.join(dev_dir, 'EXIST2025_dev.json')).T.reset_index(drop=True)

    # Standardize ID column
    tr_df = tr_df.rename({'id_EXIST': 'id'}, axis=1)
    dev_df = dev_df.rename({'id_EXIST': 'id'}, axis=1)
    tr_df['id'] = tr_df['id'].astype('Int64')
    dev_df['id'] = dev_df['id'].astype('Int64')

    # Load gold labels
    tr_gold_hard = pd.read_json(os.path.join(train_dir, 'EXIST2025_training_task1_3_gold_hard.json'))
    tr_gold_soft = pd.read_json(os.path.join(train_dir, 'EXIST2025_training_task1_3_gold_soft.json'))
    dev_gold_hard = pd.read_json(os.path.join(dev_dir, 'EXIST2025_dev_task1_3_gold_hard.json'))
    dev_gold_soft = pd.read_json(os.path.join(dev_dir, 'EXIST2025_dev_task1_3_gold_soft.json'))

    # Merge gold labels
    tr_gold_hard = tr_gold_hard.rename({'value': 'hard_label'}, axis=1).drop('test_case', axis=1)
    tr_gold_soft = tr_gold_soft.rename({'value': 'soft_label'}, axis=1).drop('test_case', axis=1)
    dev_gold_hard = dev_gold_hard.rename({'value': 'hard_label'}, axis=1).drop('test_case', axis=1)
    dev_gold_soft = dev_gold_soft.rename({'value': 'soft_label'}, axis=1).drop('test_case', axis=1)

    if merge_data:
        # Merge training and development datasets
        df = pd.concat([dev_df, tr_df], ignore_index=True, sort=False)
        df = df.merge(pd.concat([tr_gold_hard, dev_gold_hard]), how='left', on='id')
        df = df.merge(pd.concat([tr_gold_soft, dev_gold_soft]), how='left', on='id')
        df.drop(columns=['labels_task1_1', 'labels_task1_2', 'labels_task1_3', 'number_annotators'], errors='ignore', inplace=True)

        # Filter by language
        if language in ["en", "es"]:
            df = df[df['lang'] == language]

        # Analyze dataset if requested
        if analyze:
            if label_column == "hard_label":
                df = analyze_dataset(df, text_column, label_column, output_dir, "merged_hard", soft_labels=False)
            else:
                df = analyze_dataset(df, text_column, label_column, output_dir, "merged_soft", soft_labels=True)
        
        # Apply label formatting functions
        df = create_hard_labels(df)
        df = create_soft_labels(df)       
        return df

    else:
        # Separate datasets with hard and soft labels
        tr_hard_df = tr_df.merge(tr_gold_hard, how='left', on='id')
        tr_soft_df = tr_df.merge(tr_gold_soft, how='left', on='id')
        dev_hard_df = dev_df.merge(dev_gold_hard, how='left', on='id')
        dev_soft_df = dev_df.merge(dev_gold_soft, how='left', on='id')

        # Remove unnecessary columns
        for df in [tr_hard_df, tr_soft_df, dev_hard_df, dev_soft_df]:
            df.drop(columns=['labels_task1_1', 'labels_task1_2', 'labels_task1_3', 'number_annotators'], errors='ignore', inplace=True)

        tr_hard_df = create_hard_labels(tr_hard_df)
        tr_soft_df = create_soft_labels(tr_soft_df)
        dev_hard_df = create_hard_labels(dev_hard_df)
        dev_soft_df = create_soft_labels(dev_soft_df)

        # Filter by language
        if language in ["en", "es"]:
            tr_hard_df = tr_hard_df[tr_hard_df['lang'] == language]
            tr_soft_df = tr_soft_df[tr_soft_df['lang'] == language]
            dev_hard_df = dev_hard_df[dev_hard_df['lang'] == language]
            dev_soft_df = dev_soft_df[dev_soft_df['lang'] == language]

        # Analyze each dataset if requested
        if analyze:
            tr_hard_df = analyze_dataset(tr_hard_df, text_column, "hard_label", output_dir, "training_hard", soft_labels=False)
            tr_soft_df = analyze_dataset(tr_soft_df, text_column, "soft_label", output_dir, "training_soft", soft_labels=True)
            dev_hard_df = analyze_dataset(dev_hard_df, text_column, "hard_label", output_dir, "dev_hard", soft_labels=False)
            dev_soft_df = analyze_dataset(dev_soft_df, text_column, "soft_label", output_dir, "dev_soft", soft_labels=True)

        return tr_hard_df, tr_soft_df, dev_hard_df, dev_soft_df


def data_splitting(df, label_column, split_ratio=0.2, stratify_label=True):
    """Splits the dataset into training and validation sets.

    Args:
        df (pd.DataFrame): The dataset to split.
        label_column (str): The column containing multi-label data.
        split_ratio (float): The percentage of data to allocate to the validation set (default: 0.2).
        stratify_label (bool): Whether to use stratified sampling based on labels (default: True).

    Returns:
        train_df (pd.DataFrame): Training dataset.
        val_df (pd.DataFrame): Validation dataset.
    """
    print(f"\nSplitting dataset into train and validation sets (test_size={split_ratio})...")

    if stratify_label:
        print("Using stratified sampling for multi-label classification.")
        
        if label_column == "soft_label":
            # Create a new column for stratification based on the most frequent label in soft_label
            df["stratify_label"] = df[label_column].apply(lambda x: max(x, key=x.get) if isinstance(x, dict) else None)
        else:
            df["stratify_label"] = df[label_column].apply(lambda x: x[0] if isinstance(x, list) else x)
        
        label_counts = Counter()
        for labels in df["stratify_label"]:
            if isinstance(labels, list):
                label_counts.update(labels)
            else:
                label_counts[labels] += 1
        
        # Handle rare labels
        rare_labels = [label for label, count in label_counts.items() if count < 2]
        rare_indices = df["stratify_label"].apply(lambda labels: any(label in labels for label in rare_labels) if isinstance(labels, list) else labels in rare_labels)
        rare_samples = df[rare_indices]
        df = df[~rare_indices]
        
        # Perform stratified train-test split
        try:
            train_df, val_df = train_test_split(df, test_size=split_ratio, random_state=2025, stratify=df["stratify_label"])
        except ValueError:
            print("Not enough samples for stratified split, performing random split instead.")
            train_df, val_df = train_test_split(df, test_size=split_ratio, random_state=2025)
        
        # Distribute rare samples
        train_df = pd.concat([train_df, rare_samples], ignore_index=True)
    else:
        print("Performing random train-test split.")
        train_df, val_df = train_test_split(df, test_size=split_ratio, random_state=2025)
    
    print(f"Training set size: {len(train_df)}")
    print(f"Development set size: {len(val_df)}")
    return train_df, val_df


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process EXIST 2025 dataset.")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training dataset directory")
    parser.add_argument("--dev_dir", type=str, required=True, help="Path to development dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save CSV files")
    parser.add_argument("--merge_data", type=bool, default=True, help="Merge train/dev data into one file")
    parser.add_argument("--language", type=str, choices=["all", "en", "es"], default="all", help="Select language filter")
    parser.add_argument("--evaluation_type", type=str, choices=["soft", "hard"], default="hard", help="Select for soft or hard label evaluation")
    parser.add_argument("--analyze", type=bool, default=True, help="Perform dataset analysis before saving")
    parser.add_argument("--text_column", type=str, default="tweet", help="Column name for text data")
    parser.add_argument("--label_column", type=str, choices=["soft_label", "hard_label"], default="hard_label", help="Column name for label data")
    parser.add_argument("--split_data", type=bool, default=True, help="Whether to split merged dataset into train/validation")
    parser.add_argument("--split_ratio", type=float, default=0.2, help="Ratio of validation set")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.evaluation_type == "soft":
        eval = "soft"
    else:
        eval = "hard"

    if args.merge_data:
        df = load_exist2025_dataset(args.train_dir, args.dev_dir, merge_data=True, language=args.language, analyze=args.analyze, text_column=args.text_column, label_column=args.label_column)
        df.to_csv(os.path.join(output_dir, "train_dev_merged_dataset.csv"), index=False, encoding="utf-8")
        
        if args.split_data:
            train_df, val_df = data_splitting(df, args.label_column, args.split_ratio, stratify_label=True)
            
            train_output_path = os.path.join(output_dir, f"my_split_{args.language}_{args.evaluation_type}_train_dataset.csv")
            val_output_path = os.path.join(output_dir, f"my_split_{args.language}_{args.evaluation_type}_dev_dataset.csv")
            train_df.to_csv(train_output_path, index=False, encoding="utf-8")
            val_df.to_csv(val_output_path, index=False, encoding="utf-8")
            print(f"Training dataset saved: {train_output_path}")
            print(f"Dev dataset saved: {val_output_path}")
    
    else:
        tr_hard_df, tr_soft_df, dev_hard_df, dev_soft_df = load_exist2025_dataset(args.train_dir, args.dev_dir, merge_data=False, language=args.language, analyze=args.analyze, text_column=args.text_column, label_column=args.label_column)

        tr_hard_df.to_csv(os.path.join(output_dir, "training_hard_labels.csv"), index=False, encoding="utf-8")
        tr_soft_df.to_csv(os.path.join(output_dir, "training_soft_labels.csv"), index=False, encoding="utf-8")
        dev_hard_df.to_csv(os.path.join(output_dir, "dev_hard_labels.csv"), index=False, encoding="utf-8")
        dev_soft_df.to_csv(os.path.join(output_dir, "dev_soft_labels.csv"), index=False, encoding="utf-8")
