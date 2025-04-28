import os
import re
import ast
import pandas as pd
import numpy as np
import argparse
import chardet
import emoji
import html
import unicodedata
import spacy
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
    "NO",
    "IDEOLOGICAL-INEQUALITY",
    "STEREOTYPING-DOMINANCE",
    "OBJECTIFICATION",
    "SEXUAL-VIOLENCE",
    "MISOGYNY-NON-SEXUAL-VIOLENCE"
    ]


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

# python -m spacy download en_core_web_sm
# python -m spacy download es_core_news_sm
# Load English & Spanish NLP models
nlp_en = spacy.load("en_core_web_sm")
nlp_es = spacy.load("es_core_news_sm")

# Initialize Ekphrasis Text Processor (for English)
text_processor = TextPreProcessor(
    normalize=['user', 'email', 'date', 'number', 'phone'],  # Normalize users,  emails
    segmenter="twitter",  # Segment hashtags
    corrector="twitter",  # Correct common typos
    unpack_hashtags=True,
    unpack_contractions=True,
    tokenizer=SocialTokenizer(lowercase=False).tokenize  # Keep casing
)

# Spanish contractions dictionary
spanish_contractions = {
    "pa’": "para", "pal": "para el", "na’": "nada",
    "pa": "para", "pa'": "para", "d’": "de", "del’": "del",
    "q": "que", "xq": "porque", "toy": "estoy"
}

def remove_accents(text):
    """Normalize accented characters (for Spanish)."""
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def expand_contractions(text, lang):
    """Expand contractions in English & Spanish."""
    if lang == "es":
        for contraction, replacement in spanish_contractions.items():
            text = text.replace(contraction, replacement)
    return text

def emojis_to_text(sentence, lang):
    """Convert emojis into text in English or Spanish."""
    demojized = emoji.demojize(sentence, language=lang)
    return re.sub(r':[\S]+:', lambda x: x.group().replace('_', ' ').replace('-', ' ').replace(':', ''), demojized)

def remove_urls(text):
    """Remove URLs from text using regex."""
    url_pattern = r"https?://\S+|www\.\S+"  # Matches both HTTP and WWW URLs
    return re.sub(url_pattern, "", text)

def fix_html(text):
    """Convert HTML entities (e.g., &amp; -> and)."""
    return html.unescape(text)

def tokenize_text(text, lang):
    """Tokenize using spaCy (Spanish or English)."""
    nlp = nlp_es if lang == "es" else nlp_en
    return ' '.join([token.text for token in nlp(text)])

def preprocessing(text, lang):
    """Preprocess tweets in English & Spanish, handling emojis correctly per language."""
    
    # Convert emojis to text based on language
    emoji_lang = "es" if lang == "es" else "en"
    text = emojis_to_text(text, emoji_lang)

    # Fix HTML entities
    text = fix_html(text)

    # Remove URLs
    text = remove_urls(text)

    # text = re.sub(r'@[\w]+', '', text)

    # substitution of numbers, dates ...
    # text = re.sub(r'[-\+]?([0-9]+[\.:,;\\/ -])*[0-9]+', '', text) 

    # Expand contractions based on language
    if lang == "es":
        text = expand_contractions(text, lang="es")

    # Normalize accents (for Spanish only)
    if lang == "es":
        text = remove_accents(text)

    # substituition of repetition of .
    text = re.sub(r"\.\.\.\.+", r'...', text)
    
    # the use of ¿ ¡ or is unreliable on social media, what to do?
    text = re.sub(r"¡", '', text)
    text = re.sub(r"¿", '', text)

    # useful patterns and token for substitutions
    multiple_question_marks = 'TMQM'
    multiple_exclamation_marks = 'TMEM'
    mixed_exclamation_question_marks = 'TMEQM'
    
    # substitution of multiple occurrence of ! ? and !?
    text = re.sub(r"(^|[^\?!])(!(\s*!)+)([^\?!]|$)", r'\1 ' + multiple_exclamation_marks + r' \4', text)
    text = re.sub(r"(^|[^\?!])(\?(\s*\?)+)([^\?!]|$)", r'\1 ' + multiple_question_marks + r' \4', text)
    text = re.sub(r"(^|[^\?!])([!\?](\s*[!\?])+)([^\?!]|$)", r'\1 ' + mixed_exclamation_question_marks + r' \4', text)
    
    text = re.sub(multiple_question_marks, '??', text)
    text = re.sub(multiple_exclamation_marks, '!!', text)
    text = re.sub(mixed_exclamation_question_marks, '?!', text)

    # Remove excessive spaces & unwanted characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII chars
    text = re.sub(' +', ' ', text).strip()  # Normalize spaces

    # remove all form of repetition
    text = re.sub(r"([^\.]+?)\1+", r'\1\1', text)    

    # Apply Ekphrasis for tweets
    text = ' '.join(text_processor.pre_process_doc(text))

    # === Keep only one <user> token if multiple exist ===
    text = re.sub(r'(<user>\s*)+', '<user> ', text).strip()

    # add space after dots
    text = re.sub(r'([a-z])(\.|\.\.\.|\?|!|:|;|,|"|\)|}|]|…)(\w)', r'\1\2 \3', text)
     
    # remove useless spaces
    text = re.sub(r"(\s)\1*", r'\1', text)
    text = re.sub(r"(^\s*|\s*$)", r'', text)

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


def compute_label_rarity(df, label_column="hard_labels", label_names=None):
    """
    Compute inverse frequency for each label for use as rarity.

    Args:
        df (pd.DataFrame): Dataset with hard labels.
        label_column (str): Column name containing hard labels.
        label_names (list): List of label names (for ordering).

    Returns:
        dict: {label_idx: rarity_score}
    """
    from collections import Counter
    label_counts = Counter()

    for labels in df[label_column]:
        if isinstance(labels, str):
            labels = json.loads(labels)
        if isinstance(labels, dict):
            labels = list(labels.values())
        for idx, val in enumerate(labels):
            if val == 1:
                label_counts[idx] += 1

    total = len(df)
    rarity = {
        idx: 1 / (label_counts[idx] / total + 1e-6)  # avoid division by zero
        for idx in range(len(label_names))
    }
    return rarity


def compute_difficulty(row, evaluation_type="hard", rarity_scores=None):
    """
    Compute a difficulty score including label rarity (for hard/soft/both modes).
    
    Args:
        row (dict or pd.Series)
        evaluation_type (str): "hard", "soft", or "both"
        rarity_scores (dict or None): Precomputed rarity scores per label index
    
    Returns:
        float: Difficulty score
    """
    label_entropy = 0.0
    num_labels = 0
    avg_rarity = 0.0

    # --- Soft label entropy ---
    if evaluation_type in ["soft", "both"] and 'soft_labels' in row and row['soft_labels']:
        soft_labels = row['soft_labels']
        if isinstance(soft_labels, str):
            try:
                soft_labels = json.loads(soft_labels)
            except Exception:
                soft_labels = []
        if isinstance(soft_labels, dict):
            soft_labels = list(soft_labels.values())
        if isinstance(soft_labels, list) and sum(soft_labels) > 0:
            label_entropy = entropy(soft_labels)

    # --- Hard label count and rarity ---
    if evaluation_type in ["hard", "both"] and 'hard_labels' in row and row['hard_labels']:
        hard_labels = row['hard_labels']
        if isinstance(hard_labels, str):
            try:
                hard_labels = json.loads(hard_labels)
            except Exception:
                hard_labels = []
        if isinstance(hard_labels, dict):
            hard_labels = list(hard_labels.values())
        if isinstance(hard_labels, list):
            num_labels = sum(hard_labels)
            if rarity_scores:
                label_rarities = [rarity_scores.get(i, 1.0) for i, val in enumerate(hard_labels) if val == 1]
                if label_rarities:
                    avg_rarity = sum(label_rarities) / len(label_rarities)

    # --- Inverted text length ---
    tweet = row.get('tweet', "")
    text_length = len(tweet.split()) if isinstance(tweet, str) else 0
    inverted_text_score = 1 / (text_length + 1)

    # --- Final score ---
    if evaluation_type == "hard":
        difficulty = (
            0.6 * num_labels + 
            0.1 * inverted_text_score + 
            0.3 * avg_rarity
        )
    elif evaluation_type == "soft":
        difficulty = (
            0.8 * label_entropy + 
            0.1 * inverted_text_score + 
            0.1 * avg_rarity
        )
    else:  # both
        difficulty = (
            0.4 * label_entropy + 
            0.3 * num_labels + 
            0.2 * avg_rarity + 
            0.1 * inverted_text_score
        )

    return difficulty


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
    df[text_column] = df.apply(lambda row: preprocessing(row[text_column], row["lang"]), axis=1)

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


def load_exist2025_dataset(train_dir, dev_dir, merge_data=True, curriculum_learning=True, language="all", analyze=True, text_column="tweet", label_column="hard_label"):
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
            df.reset_index(drop=True, inplace=True)

        # Analyze dataset if requested
        if analyze:
            if label_column == "hard_label":
                df = analyze_dataset(df, text_column, label_column, output_dir, "merged_hard", soft_labels=False)
            else:
                df = analyze_dataset(df, text_column, label_column, output_dir, "merged_soft", soft_labels=True)
        
        # Apply label formatting functions
        df = create_hard_labels(df)
        df = create_soft_labels(df)

        if curriculum_learning:
            print("\nComputing difficulty scores and sorting dataset for Curriculum Learning...")
            df['difficulty'] = df.apply(lambda row: compute_difficulty(row, "both"), axis=1)

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
            tr_hard_df.reset_index(drop=True, inplace=True)
            tr_soft_df = tr_soft_df[tr_soft_df['lang'] == language]
            tr_soft_df.reset_index(drop=True, inplace=True)
            dev_hard_df = dev_hard_df[dev_hard_df['lang'] == language]
            dev_hard_df.reset_index(drop=True, inplace=True)
            dev_soft_df = dev_soft_df[dev_soft_df['lang'] == language]
            dev_soft_df.reset_index(drop=True, inplace=True)

        # Analyze each dataset if requested
        if analyze:
            tr_hard_df = analyze_dataset(tr_hard_df, text_column, "hard_label", output_dir, "training_hard", soft_labels=False)
            tr_soft_df = analyze_dataset(tr_soft_df, text_column, "soft_label", output_dir, "training_soft", soft_labels=True)
            dev_hard_df = analyze_dataset(dev_hard_df, text_column, "hard_label", output_dir, "dev_hard", soft_labels=False)
            dev_soft_df = analyze_dataset(dev_soft_df, text_column, "soft_label", output_dir, "dev_soft", soft_labels=True)

        if curriculum_learning:
            print("\nComputing difficulty scores and sorting datasets for Curriculum Learning...")
            tr_hard_df['difficulty'] = tr_hard_df.apply(lambda row: compute_difficulty(row, "hard"), axis=1)
            tr_soft_df['difficulty'] = tr_soft_df.apply(lambda row: compute_difficulty(row, "soft"), axis=1)
            dev_hard_df['difficulty'] = dev_hard_df.apply(lambda row: compute_difficulty(row, "hard"), axis=1)
            dev_soft_df['difficulty'] = dev_soft_df.apply(lambda row: compute_difficulty(row, "soft"), axis=1)

        return tr_hard_df, tr_soft_df, dev_hard_df, dev_soft_df


def load_and_preprocess_exist2025_test(test_dir, language="all", text_column="tweet"):
    """
    Load and preprocess EXIST2025 test set (unlabeled).
    
    Args:
        test_dir (str): Path to test dataset directory.
        language (str): "all", "en", or "es" (default: "all").
        text_column (str): Column containing the text (default: "tweet").
    
    Returns:
        test_df (pd.DataFrame): Preprocessed test DataFrame.
    """
    print("\nLoading and preprocessing EXIST2025 test set...")

    # === Load
    test_dir = os.path.abspath(test_dir)

    # Load test data
    test_df = pd.read_json(os.path.join(test_dir, 'EXIST2025_test_clean.json')).T.reset_index(drop=True)

    # Standardize ID column
    test_df = test_df.rename({'id_EXIST': 'id'}, axis=1)
    test_df.drop(columns=['number_annotators', 'annotators', 'gender_annotators', 'age_annotators', 'ethnicities_annotators', 'study_levels_annotators', 'countries_annotators'], errors='ignore', inplace=True)

    # Filter by language
    if language in ["en", "es"]:
        test_df = test_df[test_df['lang'] == language]
        test_df.reset_index(drop=True, inplace=True)

    # === Preprocessing
    print("Preprocessing tweets...")
    test_df[text_column] = test_df.apply(lambda row: preprocessing(row[text_column], row["lang"]), axis=1)

    print(f"\nPerforming Sentiment Analysis on unlabelled test dataset...")
    test_df['sentiment_score'] = test_df[text_column].astype(str).apply(analyze_sentiment)
    test_df['sentiment'] = test_df['sentiment_score'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

    # Save Sentiment Distribution Plot
    print(f"\nSaving Sentiment Distribution for unlabelled test dataset...")
    save_sentiment_distribution(test_df, os.path.join(output_dir, f"test_sentiment_distribution.png"))

    return test_df


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
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save CSV files")
    parser.add_argument("--merge_data", type=bool, default=True, help="Merge train/dev data into one file")
    parser.add_argument("--language", type=str, choices=["all", "en", "es"], default="all", help="Select language filter")
    parser.add_argument("--evaluation_type", type=str, choices=["soft", "hard"], default="hard", help="Select for soft or hard label evaluation")
    parser.add_argument("--analyze", type=bool, default=True, help="Perform dataset analysis before saving")
    parser.add_argument("--text_column", type=str, default="tweet", help="Column name for text data")
    parser.add_argument("--label_column", type=str, choices=["soft_label", "hard_label"], default="hard_label", help="Column name for label data")
    parser.add_argument("--split_data", type=bool, default=False, help="Whether to split merged dataset into train/validation")
    parser.add_argument("--split_ratio", type=float, default=0.2, help="Ratio of validation set")
    parser.add_argument("--curriculum_learning", type=bool, default=True, help="Compute difficulty scores for curriculum learning")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess training and development sets
    if args.merge_data:
        df = load_exist2025_dataset(args.train_dir, args.dev_dir, merge_data=True, curriculum_learning = args.curriculum_learning, language=args.language, analyze=args.analyze, text_column=args.text_column, label_column=args.label_column)
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
        tr_hard_df, tr_soft_df, dev_hard_df, dev_soft_df = load_exist2025_dataset(args.train_dir, args.dev_dir, merge_data=False, curriculum_learning = args.curriculum_learning, language=args.language, analyze=args.analyze, text_column=args.text_column, label_column=args.label_column)

        tr_hard_df.to_csv(os.path.join(output_dir, "training_hard_labels.csv"), index=False, encoding="utf-8")
        tr_soft_df.to_csv(os.path.join(output_dir, "training_soft_labels.csv"), index=False, encoding="utf-8")
        dev_hard_df.to_csv(os.path.join(output_dir, "dev_hard_labels.csv"), index=False, encoding="utf-8")
        dev_soft_df.to_csv(os.path.join(output_dir, "dev_soft_labels.csv"), index=False, encoding="utf-8")

    # Load and preprocess test set
    test_df = load_and_preprocess_exist2025_test(test_dir = args.test_dir,language = args.language, text_column = args.text_column)
    test_df.to_csv(os.path.join(output_dir, "test_no_labels_preprocessed.csv"), index=False, encoding="utf-8")
    print("Dataset Preprocessing completed.")
