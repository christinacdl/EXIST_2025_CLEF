import os
import numpy as np
import pandas as pd
import torch
import ast
import json
import random
from scipy.special import expit
import csv
import math
from skimpy import skim
import traceback
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from typing import Literal,  Dict, Any
import tempfile
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from transformers import Trainer, DataCollatorWithPadding
from transformers.trainer_utils import enable_full_determinism
from some_loss import ResampleLoss
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, multilabel_confusion_matrix
)
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils

# Set up GPU for Training
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Label mapping
LABEL_LIST = [
    "NO",
    "IDEOLOGICAL-INEQUALITY",
    "STEREOTYPING-DOMINANCE",
    "OBJECTIFICATION",
    "SEXUAL-VIOLENCE",
    "MISOGYNY-NON-SEXUAL-VIOLENCE"
    ]

id2label = {i: label for i, label in enumerate(LABEL_LIST)}
label2id = {label: i for i, label in enumerate(LABEL_LIST)}
#################################################################################################################


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    enable_full_determinism(seed_value)


# Function to check if a directory exists else creates the directory
def check_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Directory created at {}'.format(path))
    else:
        print('Directory {} already exists!'.format(path))


# Function to write a DataFrame to a TSV file
def write_tsv_dataframe(filepath, dataframe):
    try:
        dataframe.to_csv(filepath, encoding="utf-8", sep="\t", index=False, header=True, quoting=csv.QUOTE_NONE)
    except IOError:
        traceback.print_exc()


# Convert stringified lists into actual lists and ensure they are floats
def safe_literal_eval(x):
    try:
        if isinstance(x, str):
            return np.array(ast.literal_eval(x), dtype=float)
        else:
            return np.array(x, dtype=float)
    except (ValueError, SyntaxError) as e:
        print(f"Error converting {x}: {e}")
        return np.array([], dtype=float)


# Function to load and process data
def process_dataset(
    file_path,
    evaluation_type,
    label_column,
    text_column,
    sentiment_column,
    language,
    tokenizer,
    baseline=True,
    sentiment=True,
    curriculum_learning=True
):
    """
    Load and process a dataset for multi-label sexism and sentiment classification.
    Handles both labeled and unlabeled test sets.

    Args:
        file_path (str): Path to dataset CSV.
        evaluation_type (str): "soft" or "hard" label type (unused here, but passed for compatibility).
        label_column (str or None): Column name for multi-label sexism labels.
        text_column (str): Column name for tweet text.
        sentiment_column (str or None): Column name for sentiment labels.
        language (str): "en", "es", or "both".
        tokenizer: Tokenizer from HuggingFace.
        sentiment (bool): Whether sentiment labels are expected.
        curriculum_learning (bool): Whether difficulty-based sorting is applied.

    Returns:
        Dataset object, cleaned DataFrame(s), max length
    """
    df = pd.read_csv(file_path)

    # Filter by language
    if language in ["en", "es"]:
        df = df[df["lang"] == language]
        df.reset_index(drop=True, inplace=True)

    has_labels = label_column in df.columns and not df[label_column].isnull().all()
    has_sentiment = sentiment and sentiment_column in df.columns and not df[sentiment_column].isnull().all()
    has_difficulty = "difficulty" in df.columns

    # === Label processing ===
    if has_labels:
        if baseline:
            df["labels"] = df[label_column].apply(safe_literal_eval)
        else:
            df["labels_sexist"] = df[label_column].apply(safe_literal_eval)

    if has_sentiment:
        sentiment_mapping = {"Negative": 0, "Neutral": 1, "Positive": 2}
        df["labels_sentiment"] = df[sentiment_column].map(sentiment_mapping)

    # === Select Columns ===
    keep_cols = ["id", text_column, "lang"]

    if has_labels:
        if baseline:
            keep_cols.append("labels")
        else:
            keep_cols.append("labels_sexist")
    if has_sentiment:
        keep_cols.append("labels_sentiment")
    if curriculum_learning and has_difficulty:
        keep_cols.append("difficulty")

    df = df[keep_cols]

    # === Skim summary ===
    skimpy_file = skim(df)
    print(skimpy_file)
    
    # === Compute max length ===
    encoded_texts = [tokenizer.encode(sent, add_special_tokens=True) for sent in df[text_column].values]
    max_len = max(len(sent) for sent in encoded_texts)
    print("\nMaximum sequence length:", max_len)

    # === Curriculum sorting if applicable ===
    if curriculum_learning and has_difficulty:
        cl_df = df.sort_values(by="difficulty", ascending=True).reset_index(drop=True)
        dataset = Dataset.from_pandas(cl_df)
        return dataset, cl_df, df, max_len
    else:
        dataset = Dataset.from_pandas(df)
        return dataset, df, max_len


# Tokenization function
def tokenize(examples, column, tokenizer, max_len, args):
    return tokenizer(
            examples[column], 
            padding=args.padding, 
            max_length= max_len, 
            truncation=args.truncation, 
            add_special_tokens=args.add_special_tokens, 
            return_attention_mask=args.return_attention_mask, 
            return_tensors="pt"
        )


# Tokenization function for multi-task learning
lang2id = {"en": 0, "es": 1}
def tokenize_sentiment_sexism(examples, column, tokenizer, max_len, args):
    """
    Tokenizes text and adds sentiment and sexism labels, along with language.
    
    Args:
        example (dict): A single row from the dataset.
        text_column (str): Column name containing the text.
        tokenizer: HuggingFace tokenizer.
        max_len (int): Maximum token length.
        args: Argparse arguments with padding and truncation info.

    Returns:
        dict: Tokenized input with labels and language field.
    """
    tokenized = tokenizer(
        examples[column],
        padding=args.padding,
        truncation=args.truncation,
        max_length=max_len
    )
    if "labels_sexist" in examples:
        tokenized["labels_sexist"] = examples["labels_sexist"]
    tokenized["labels_sentiment"] = examples["labels_sentiment"]
    tokenized["language"] = [lang2id[lang] for lang in examples["lang"]]
    return tokenized


# Function to compute positive weights for BCEWithLogitsLoss
def compute_positive_weights(df, label_list, label_column="labels"):
    """
    Compute pos_weight for BCEWithLogitsLoss: total_neg / total_pos for each class.
    """
    label_counts = np.zeros(len(label_list))

    for labels in df[label_column]:
        label_counts += np.array(labels)

    total_samples = len(df)
    total_negatives = total_samples - label_counts
    total_positives = label_counts

    # Avoid division by 0
    pos_weights = np.where(total_positives > 0, total_negatives / total_positives, 1.0)

    print("Class Positive Weights: ", pos_weights.tolist())
    return pos_weights.tolist()


# Function to compute positive weights for multi-label classification by language
def compute_positive_weights_lang(df, label_list, label_column="labels"):
    pos_weights = {}
    for lang in ["en", "es"]:
        lang_df = df[df["lang"] == lang]
        if lang_df.empty:
            continue

        label_counts = np.zeros(len(label_list))
        for labels in lang_df[label_column]:
            label_counts += np.array(labels)

        total_samples = len(lang_df)
        total_negatives = total_samples - label_counts
        total_positives = label_counts

        weights = np.where(total_positives > 0, total_negatives / total_positives, 1.0)
        pos_weights[lang] = torch.tensor(weights, dtype=torch.float32).to(device)
        print(f"[{lang}] Pos Weights:", weights)

    return pos_weights


# Function to compute the class frequencies for multi-label classification
def compute_class_frequencies(df, label_list, label_column="labels_sexist"):
    """
    Compute class frequencies for multi-label classification.
    
    Args:
        df (pd.DataFrame): DataFrame containing binary labels.
        label_list (list): List of label names.
        
    Returns:
        torch.Tensor: Class frequencies tensor.
    """
    label_counts = Counter()
    
    for label_vector in df[label_column]:
        for i, label_value in enumerate(label_vector):
            if label_value == 1:
                label_counts[label_list[i]] += 1
    
    class_freq = [label_counts[label] for label in label_list]
    class_freq_tensor = torch.tensor(class_freq, dtype=torch.float).to(device)
    
    print("Class frequencies:", class_freq)
    return class_freq_tensor


# Function to compute the number of training samples
def compute_train_number(df):
    """
    Compute the number of training samples.
    
    Args:
        df (pd.DataFrame): DataFrame containing training data.
        
    Returns:
        torch.Tensor: Train number tensor.
    """
    train_num = len(df)
    print(f"Total training samples: {train_num}")
    return torch.tensor(train_num, dtype=torch.float).to(device)


# Function to flatten a nested dictionary
def flatten_dict(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key to use for the flattened keys.
        sep (str): The separator to use between keys.

    Returns:
        dict: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class MultiLabelTrainer(Trainer):
    def __init__(self, model, args, train_dataset=None, eval_dataset=None, 
                compute_metrics=None, class_weights=None, evaluation_type = 'hard', loss_func_name: Literal['FL', 'CBloss', 'DBloss', 'CBloss-ntr', 'BCE'] = 'BCE', 
                class_frequencies=None, train_number=None, data_collator=None, **kwargs):
        """
        Trainer for Multi-Label Classification with Curriculum Learning
        - Supports BCE, Focal Loss, and Class-Balanced Loss.
        - Uses a dynamic difficulty scheduler to control data complexity over epochs.
        """
        super().__init__(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
                        compute_metrics=compute_metrics, data_collator=data_collator, **kwargs)
        
        self.class_weights = class_weights
        self.class_freq = class_frequencies
        self.train_num = train_number
        self.loss_func_name = loss_func_name
        self.evaluation_type = evaluation_type

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Computes the loss for multi-label classification.

        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.loss_func_name == 'FL':
            # FOCAL LOSS
            loss_fn = ResampleLoss(reweight_func=None, loss_weight=1.0,
                                    focal=dict(focal=True, alpha=0.5, gamma=2),
                                    logit_reg=dict(), class_freq=self.class_freq, train_num=self.train_num)
            
        elif self.loss_func_name == 'CBloss': 
            # CLASS-BALANCED LOSS
            loss_fn = ResampleLoss(reweight_func='CB', loss_weight=5.0,
                                    focal=dict(focal=True, alpha=0.5, gamma=2),
                                    logit_reg=dict(),
                                    CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                                    class_freq=self.class_freq, train_num=self.train_num)

        elif self.loss_func_name == 'DBloss': 
            # DISTRIBUTION-BALANCED LOSS
            loss_fn = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                                    focal=dict(focal=True, alpha=0.5, gamma=2),
                                    logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                    map_param=dict(alpha=0.1, beta=10.0, gamma=0.05), 
                                    class_freq=self.class_freq, train_num=self.train_num)
            
        elif self.loss_func_name == 'CBloss-ntr': 
            # CLASS-BALANCED NEGATIVE TOLERANT REGULARIZATION LOSS
            loss_fn = ResampleLoss(reweight_func='CB', loss_weight=10.0,
                                    focal=dict(focal=True, alpha=0.5, gamma=2),
                                    logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                    CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                                    class_freq=self.class_freq, train_num=self.train_num)
  
        elif self.loss_func_name == 'BCE':
            # BINARY CROSS-ENTROPY LOSS WITH LOGITS
            if self.class_weights is not None:
                class_weights_tensor = torch.tensor(self.class_weights, device=labels.device, dtype=logits.dtype)
                loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)
            else:
                loss_fn = torch.nn.BCEWithLogitsLoss()

        else:
            raise ValueError(f"Unknown loss_func_name: {self.loss_func_name}")        
        
        if self.evaluation_type == 'hard':
            loss = loss_fn(logits, labels.float())        
        elif self.evaluation_type == 'soft':    
            loss = loss_fn(logits, labels)
        else:
            raise ValueError(f"Unknown evaluation_type: {self.evaluation_type}")          

        return (loss, outputs) if return_outputs else loss 


class DataCollatorMultiTask:
    def __init__(self, tokenizer, padding=True, max_length=None, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.base_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of
        )

    def __call__(self, features: list[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        
        # Handle missing labels gracefully
        labels_sexist = [f.pop("labels_sexist", None) for f in features]
        labels_sentiment = [f.pop("labels_sentiment", None) for f in features]
        language_ids = [f.pop("language", None) for f in features]

        # Use base tokenizer-aware collator
        batch = self.base_collator(features)

        # Add labels only if they exist
        if all(label is not None for label in labels_sexist):
            batch["labels_sexist"] = torch.tensor(np.array(labels_sexist), dtype=torch.float32)
        if all(label is not None for label in labels_sentiment):
            batch["labels_sentiment"] = torch.tensor(labels_sentiment, dtype=torch.long)
        if all(lang is not None for lang in language_ids):
            batch["language"] = torch.tensor(language_ids, dtype=torch.long)

        return batch


class MultiTaskTrainer(Trainer):
    def __init__(self, model, args, train_dataset=None, eval_dataset=None, 
                 compute_metrics=None, class_weights=None, evaluation_type='hard',
                 loss_func_name: Literal['FL', 'CBloss', 'DBloss', 'CBloss-ntr', 'BCE'] = 'BCE', 
                 class_frequencies=None, train_number=None, data_collator=None, **kwargs):
        
        super().__init__(model=model, args=args, train_dataset=train_dataset, 
                         eval_dataset=eval_dataset, compute_metrics=compute_metrics, 
                         data_collator=data_collator, **kwargs)
        
        self.class_weights = class_weights
        self.class_freq = class_frequencies
        self.train_num = train_number
        self.loss_func_name = loss_func_name
        self.evaluation_type = evaluation_type

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # === Extract multitask labels & language ===
        labels_sexist = inputs.pop("labels_sexist")
        labels_sentiment = inputs.pop("labels_sentiment")
        language = inputs.pop("language")

        # === Forward pass ===
        outputs = model(
            **inputs,
            labels_sexist=labels_sexist,
            labels_sentiment=labels_sentiment,
            language=language,
        )
        logits_sexist, logits_sentiment, _ = outputs.logits

        # print("=== Debug Info ===")
        # print("logits_sexist:", torch.isnan(logits_sexist).any(), logits_sexist.shape)
        # print("labels_sexist:", torch.isnan(labels_sexist).any(), labels_sexist.shape, labels_sexist.dtype)

        # print("logits_sentiment:", torch.isnan(logits_sentiment).any(), logits_sentiment.shape)
        # print("labels_sentiment:", torch.isnan(labels_sentiment).any(), labels_sentiment.shape, labels_sentiment.dtype)

        # === Multi-label sexism loss ===
        if self.loss_func_name == 'FL':
            loss_fn_sexist = ResampleLoss(
                reweight_func=None, loss_weight=1.0,
                focal=dict(focal=True, alpha=0.5, gamma=2),
                logit_reg=dict(), class_freq=self.class_freq, train_num=self.train_num
            )
        elif self.loss_func_name == 'CBloss':
            loss_fn_sexist = ResampleLoss(
                reweight_func='CB', loss_weight=5.0,
                focal=dict(focal=True, alpha=0.5, gamma=2),
                logit_reg=dict(), CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                class_freq=self.class_freq, train_num=self.train_num
            )
        elif self.loss_func_name == 'DBloss':
            loss_fn_sexist = ResampleLoss(
                reweight_func='rebalance', loss_weight=1.0,
                focal=dict(focal=True, alpha=0.5, gamma=2),
                logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                map_param=dict(alpha=0.1, beta=10.0, gamma=0.05),
                class_freq=self.class_freq, train_num=self.train_num
            )
        elif self.loss_func_name == 'CBloss-ntr':
            loss_fn_sexist = ResampleLoss(
                reweight_func='CB', loss_weight=10.0,
                focal=dict(focal=True, alpha=0.5, gamma=2),
                logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                class_freq=self.class_freq, train_num=self.train_num
            )
        elif self.loss_func_name == 'BCE':
            if isinstance(self.class_weights, dict):
                lang_id = language[0].item() if isinstance(language, torch.Tensor) else language[0]
                lang = 'en' if lang_id == 0 else 'es' 
                class_weights_tensor = torch.tensor(
                    self.class_weights[lang], dtype=logits_sexist.dtype, device=logits_sexist.device)
            elif isinstance(self.class_weights, torch.Tensor):
                class_weights_tensor = self.class_weights.to(device=logits_sexist.device, dtype=logits_sexist.dtype)
            else:
                class_weights_tensor = None
            
            loss_fn_sexist = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor) \
                if class_weights_tensor is not None else nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss_func_name: {self.loss_func_name}")

        # === Final sexism loss ===
        loss_sexist = loss_fn_sexist(
            logits_sexist,
            labels_sexist.float() if self.evaluation_type == "hard" else labels_sexist
        )

        # === Sentiment loss (CrossEntropy for multi-class) ===
        loss_sentiment = nn.CrossEntropyLoss()(logits_sentiment, labels_sentiment)

        # === Final combined loss ===
        loss = loss_sexist + 0.3 * loss_sentiment  # 0.3 best

        return (loss, outputs) if return_outputs else loss


#ICM UTILS
# Function to get all parents of a label in the hierarchy
def get_all_parents(label, hierarchy, path=None):
    if path is None:
        path = []
    for parent, children in hierarchy.items():
        if isinstance(children, dict):
            if label in children:
                return path + [parent]
            res = get_all_parents(label, children, path + [parent])
            if res:
                return res
        elif isinstance(children, list):
            if label in children:
                return path + [parent]
    return []


# Function to compute label frequencies
def compute_label_frequencies(y_true, label_list, hierarchy):
    freq = {label: 0 for label in label_list}
    for parent in hierarchy.keys():
        if parent not in freq:
            freq[parent] = 0

    for row in y_true:
        for i, v in enumerate(row):
            if v == 1:
                label = label_list[i]
                freq[label] += 1
                for parent in get_all_parents(label, hierarchy):
                    if parent in freq:
                        freq[parent] += 1

    total = sum(freq.values())
    return {label: freq[label] / total if total > 0 else 1e-12 for label in freq}


# Function to compute information content
def compute_information_content(label_set, label_probs):
    return sum([-math.log2(label_probs.get(label, 1e-12)) for label in label_set]) if label_set else 0


# Function to compute ICM score
def compute_icm_score(y_pred_bin, y_true_bin, label_list, label_probs, hierarchy, alpha1=2, alpha2=2, beta=3):
    total_score = 0
    for pred_row, true_row in zip(y_pred_bin, y_true_bin):
        pred_labels = [label_list[i] for i, v in enumerate(pred_row) if v == 1]
        true_labels = [label_list[i] for i, v in enumerate(true_row) if v == 1]
        union_labels = list(set(pred_labels) | set(true_labels))
        score = (
            alpha1 * compute_information_content(pred_labels, label_probs) +
            alpha2 * compute_information_content(true_labels, label_probs) -
            beta * compute_information_content(union_labels, label_probs)
        )
        total_score += score
    return total_score / len(y_true_bin)


# Function to compute F1 score using PyEvALL's generalized formula
def f1_pyevall_style(tp, pred_count, gold_count, alpha=0.5):
    """
    Computes F1 using PyEvALL's generalized formula with a tunable alpha.
    """
    if pred_count == 0 or gold_count == 0:
        return 0.0

    precision = tp / pred_count
    recall = tp / gold_count

    if precision == 0 or recall == 0:
        return 0.0

    return 1 / ((alpha / precision) + ((1 - alpha) / recall))


# Function to compute metrics for multi-label classification
def compute_metrics(eval_pred, NUM_LABELS, label_list, thresholds=np.arange(0.1, 1.0, 0.05)):
    hierarchy = {
        "YES": ["IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE", "OBJECTIFICATION",
                "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE"],
        "NO": []
    }

    logits, labels = eval_pred
    y_true = labels
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits)).numpy()

    best_thresholds = {label: 0.5 for label in label_list}
    best_f1_scores = {label: 0.0 for label in label_list}
    macro_f1_per_threshold = {}

    label_probs = compute_label_frequencies(y_true, label_list, hierarchy)
    icm_scores = {}

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        f1s = []

        for i, label in enumerate(label_list):
            tp = ((preds[:, i] == 1) & (y_true[:, i] == 1)).sum()
            pred_count = preds[:, i].sum()
            gold_count = y_true[:, i].sum()

            f1 = f1_pyevall_style(tp, pred_count, gold_count)
            f1s.append(f1)

            if f1 > best_f1_scores[label]:
                best_f1_scores[label] = f1
                best_thresholds[label] = threshold

        macro_f1 = np.mean(f1s)
        macro_f1_per_threshold[threshold] = macro_f1

        icm = compute_icm_score(preds, y_true, label_list, label_probs, hierarchy)
        icm_scores[threshold] = icm

    best_threshold = max(macro_f1_per_threshold, key=macro_f1_per_threshold.get)
    best_icm_threshold = max(icm_scores, key=icm_scores.get)

    return {
        "F1_macro": round(macro_f1_per_threshold[best_threshold], 4),
        "ICM": round(icm_scores[best_icm_threshold], 4),
        "best_threshold": best_threshold,
        "per_class": {
            label: {
                "threshold": best_thresholds[label],
                "FMeasure": round(best_f1_scores[label], 4)
            } for label in label_list
        }
    }


# Function to compute metrics for multi-task learning
def compute_metrics_multitask(eval_pred, label_list, thresholds=np.arange(0.1, 1.0, 0.05)):
    """
    Compute metrics for multi-task setup: multi-label sexism + multi-class sentiment.
    """

    hierarchy = {
        "YES": ["IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE", "OBJECTIFICATION",
                "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE"],
        "NO": []
    }

    id2lang = {0: "en", 1: "es"}

    logits = eval_pred.predictions
    if isinstance(logits, dict):
        logits_sexist, logits_sentiment = logits["logits"]
        language_ids = logits.get("language")
    elif isinstance(logits, tuple) and len(logits) == 3:
        logits_sexist, logits_sentiment, language_ids = logits
    else:
        raise ValueError("Expected predictions to include language IDs (logits_sexist, logits_sentiment, language_ids)")

    labels = eval_pred.label_ids
    if isinstance(labels, tuple) and len(labels) == 2:
        labels_sexist, labels_sentiment = labels
    else:
        raise ValueError("Expected label_ids to be a tuple of (labels_sexist, labels_sentiment)")

    preds_sentiment = np.argmax(logits_sentiment, axis=1)
    sentiment_acc = accuracy_score(labels_sentiment, preds_sentiment)

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.tensor(logits_sexist)).numpy()

    label_probs = compute_label_frequencies(labels_sexist, label_list, hierarchy)
    best_thresholds = {label: 0.5 for label in label_list}
    best_f1_scores = {label: 0.0 for label in label_list}
    macro_f1_per_threshold = {}
    icm_scores = {}
    icm_by_lang = defaultdict(list)

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        f1s = []

        for i, label in enumerate(label_list):
            tp = ((preds[:, i] == 1) & (labels_sexist[:, i] == 1)).sum()
            pred_count = preds[:, i].sum()
            gold_count = labels_sexist[:, i].sum()
            f1 = f1_pyevall_style(tp, pred_count, gold_count)
            f1s.append(f1)

            if f1 > best_f1_scores[label]:
                best_f1_scores[label] = f1
                best_thresholds[label] = threshold

        macro_f1 = np.mean(f1s)
        macro_f1_per_threshold[threshold] = macro_f1

        icm = compute_icm_score(preds, labels_sexist, label_list, label_probs, hierarchy)
        icm_scores[threshold] = icm

        if language_ids is not None:
            language_ids_np = np.array(language_ids)
            for lang_id in np.unique(language_ids_np):
                indices = np.where(language_ids_np == lang_id)[0]
                lang = id2lang.get(int(lang_id), f"lang_{lang_id}")
                if len(indices) > 0:
                    icm_lang = compute_icm_score(preds[indices], labels_sexist[indices], label_list, label_probs, hierarchy)
                    icm_by_lang[lang].append(icm_lang)

    best_icm_threshold = max(icm_scores, key=icm_scores.get)
    best_icm_score = icm_scores[best_icm_threshold]

    return {
        "ICM": round(best_icm_score, 4),
        "F1_macro": round(macro_f1_per_threshold[best_icm_threshold], 4),
        "Sentiment_Accuracy": round(sentiment_acc, 4),
        "best_threshold": best_icm_threshold,
        "ICM_per_language": {lang: round(np.mean(scores), 4) for lang, scores in icm_by_lang.items()},
        "per_class": {
            label: {
                "threshold": best_thresholds[label],
                "FMeasure": round(best_f1_scores[label], 4)
            } for label in label_list
        }
    }


def find_best_thresholds(predictions, labels, dataframe, label_list, output_json, thresholds=np.arange(0.1, 1.0, 0.05)):
    """
    Finds the best per-class thresholds and computes PyEvALL-style F1 metrics.

    Args:
        predictions (np.ndarray): Logits.
        labels (np.ndarray): Ground truth.
        dataframe (pd.DataFrame): Contains IDs (unused here but kept for compatibility).
        label_list (list): List of class labels.
        output_json (str): Where to save the threshold config.
        thresholds (np.array): Range of thresholds.

    Returns:
        metrics (dict): Contains best thresholds per class and macro F1.
        best_threshold (float): The overall best threshold (based on average F1).
    """
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.tensor(predictions)).numpy()
    y_true = labels

    best_per_class = {label: {"F1": 0.0, "threshold": 0.5} for label in label_list}
    macro_f1_per_threshold = {}
    
    for threshold in thresholds:
        bin_preds = (probs >= threshold).astype(int)

        per_class_f1 = []
        for i, label in enumerate(label_list):
            tp = ((bin_preds[:, i] == 1) & (y_true[:, i] == 1)).sum()
            pred_count = bin_preds[:, i].sum()
            gold_count = y_true[:, i].sum()

            f1 = f1_pyevall_style(tp, pred_count, gold_count)
            per_class_f1.append(f1)

            if f1 > best_per_class[label]["F1"]:
                best_per_class[label]["F1"] = f1
                best_per_class[label]["threshold"] = float(threshold)

        macro_f1_per_threshold[float(threshold)] = np.mean(per_class_f1)

    # Get best general threshold based on highest macro F1
    best_overall_threshold = max(macro_f1_per_threshold, key=macro_f1_per_threshold.get)
    best_macro_f1 = macro_f1_per_threshold[best_overall_threshold]

    # Save metrics in compatible format
    results = {
        "per_class": best_per_class,
        "best_overall_threshold": best_overall_threshold,
        "best_overall_FMeasure": best_macro_f1
    }

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    return results, best_overall_threshold



# def find_best_thresholds_multitask(
#     sexist_logits: np.ndarray,
#     sexist_labels: np.ndarray,
#     dataframe: pd.DataFrame,
#     label_list: list,
#     output_json: str,
#     thresholds=np.arange(0.1, 1.0, 0.05)
# ):
#     """
#     Find best per-class thresholds for the multi-label sexism task.

#     Args:
#         sexist_logits (np.ndarray): Raw logits for sexism (shape: [N, C])
#         sexist_labels (np.ndarray): Ground truth labels for sexism (shape: [N, C])
#         dataframe (pd.DataFrame): Dev dataframe (for future use or debugging)
#         label_list (list): List of label names
#         output_json (str): Path to save the best thresholds and F1s
#         thresholds (np.array): Threshold range to search

#     Returns:
#         dict: metrics (best F1, thresholds)
#         float: best overall threshold
#     """
#     # Sanity check
#     if sexist_logits.ndim != 2 or sexist_labels.ndim != 2:
#         raise ValueError(f"Expected 2D arrays. Got logits: {sexist_logits.shape}, labels: {sexist_labels.shape}")

#     probs = torch.sigmoid(torch.tensor(sexist_logits)).numpy()
#     y_true = sexist_labels

#     best_per_class = {label: {"F1": 0.0, "threshold": 0.5} for label in label_list}
#     macro_f1_per_threshold = {}

#     for threshold in thresholds:
#         bin_preds = (probs >= threshold).astype(int)
#         per_class_f1 = []

#         for i, label in enumerate(label_list):
#             tp = ((bin_preds[:, i] == 1) & (y_true[:, i] == 1)).sum()
#             pred_count = bin_preds[:, i].sum()
#             gold_count = y_true[:, i].sum()

#             f1 = f1_pyevall_style(tp, pred_count, gold_count)
#             per_class_f1.append(f1)

#             if f1 > best_per_class[label]["F1"]:
#                 best_per_class[label]["F1"] = f1
#                 best_per_class[label]["threshold"] = float(threshold)

#         macro_f1_per_threshold[float(threshold)] = np.mean(per_class_f1)

#     best_overall_threshold = max(macro_f1_per_threshold, key=macro_f1_per_threshold.get)
#     best_macro_f1 = macro_f1_per_threshold[best_overall_threshold]

#     results = {
#         "per_class": best_per_class,
#         "best_overall_threshold": best_overall_threshold,
#         "best_overall_FMeasure": round(best_macro_f1, 4),
#     }

#     with open(output_json, "w") as f:
#         json.dump(results, f, indent=4)

#     return results, best_overall_threshold




# def find_best_thresholds_multitask(
#     sexist_logits: np.ndarray,
#     sentiment_logits: np.ndarray,
#     sexist_labels: np.ndarray,
#     dataframe: pd.DataFrame,
#     label_list: list,
#     output_json: str,
#     thresholds=np.arange(0.1, 1.0, 0.05),
#     sentiment_label_map={0: "Positive", 1: "Neutral", 2: "Negative"},
# ):
#     """
#     ICM-aware threshold optimization using sentiment predictions.

#     Args:
#         sexist_logits (np.ndarray): Raw logits for sexism (N x C)
#         sentiment_logits (np.ndarray): Raw logits for sentiment (N x 3)
#         sexist_labels (np.ndarray): Ground truth binary labels (N x C)
#         dataframe (pd.DataFrame): Original dev dataframe
#         label_list (list): Class names for sexism
#         output_json (str): Where to save results
#         thresholds (np.array): Thresholds to test
#         sentiment_label_map (dict): Maps sentiment index to label

#     Returns:
#         dict: thresholds + F1s
#         float: best static threshold
#     """

#     probs = torch.sigmoid(torch.tensor(sexist_logits)).numpy()
#     y_true = sexist_labels

#     sentiment_preds = np.argmax(sentiment_logits, axis=1)
#     sentiment_softmax = torch.softmax(torch.tensor(sentiment_logits), dim=-1).numpy()

#     best_per_class = {label: {"F1": 0.0, "threshold": 0.5} for label in label_list}
#     macro_f1_per_threshold = {}

#     for base_threshold in thresholds:
#         adjusted_preds = np.zeros_like(probs)

#         for i in range(probs.shape[0]):
#             sentiment_class = sentiment_label_map[sentiment_preds[i]]
#             thresholds_i = np.ones(len(label_list)) * base_threshold

#             if sentiment_class == "Negative":
#                 thresholds_i[1:] -= 0.1  # More permissive on YES subtypes
#                 thresholds_i[1:] = np.clip(thresholds_i[1:], 0.1, 0.5)
#             elif sentiment_class == "Neutral":
#                 thresholds_i[0] -= 0.1  # Bias toward NO
#                 thresholds_i[1:] += 0.1
#                 thresholds_i = np.clip(thresholds_i, 0.1, 0.9)
#             elif sentiment_class == "Positive":
#                 thresholds_i += 0.1  # More conservative overall
#                 thresholds_i = np.clip(thresholds_i, 0.5, 0.9)

#             adjusted_preds[i] = (probs[i] >= thresholds_i).astype(int)

#         per_class_f1 = []
#         for j, label in enumerate(label_list):
#             tp = ((adjusted_preds[:, j] == 1) & (y_true[:, j] == 1)).sum()
#             pred_count = adjusted_preds[:, j].sum()
#             gold_count = y_true[:, j].sum()

#             f1 = f1_pyevall_style(tp, pred_count, gold_count)
#             per_class_f1.append(f1)

#             if f1 > best_per_class[label]["F1"]:
#                 best_per_class[label]["F1"] = f1
#                 best_per_class[label]["threshold"] = float(base_threshold)

#         macro_f1_per_threshold[float(base_threshold)] = np.mean(per_class_f1)

#     best_overall_threshold = max(macro_f1_per_threshold, key=macro_f1_per_threshold.get)
#     best_macro_f1 = macro_f1_per_threshold[best_overall_threshold]

#     results = {
#         "per_class": best_per_class,
#         "best_overall_threshold": best_overall_threshold,
#         "best_overall_FMeasure": round(best_macro_f1, 4),
#     }

#     with open(output_json, "w") as f:
#         json.dump(results, f, indent=4)

#     return results, best_overall_threshold


# Function to find best thresholds for multi-task learning
def find_best_thresholds_multitask(
    sexist_logits: np.ndarray,
    sentiment_logits: np.ndarray,
    sexist_labels: np.ndarray,
    dataframe: pd.DataFrame,
    label_list: list,
    output_json: str,
    thresholds=np.arange(0.1, 1.0, 0.05),
    sentiment_label_map={0: "Positive", 1: "Neutral", 2: "Negative"},
):
    """
    ICM-aware threshold optimization using sentiment predictions + per-language class thresholds.

    Returns:
        - dict with best thresholds + F1s
        - float: best static threshold
    """

    # === Init ===
    probs = torch.sigmoid(torch.tensor(sexist_logits)).numpy()
    y_true = sexist_labels
    sentiment_preds = np.argmax(sentiment_logits, axis=1)
    lang_ids = dataframe["lang"].values  # e.g. "en", "es"

    best_per_class = {label: {"F1": 0.0, "threshold": 0.5} for label in label_list}
    macro_f1_per_threshold = {}

    # === New: store best thresholds per lang and class ===
    langs = ["en", "es"]
    best_thresh_per_lang_class = {
        lang: {label: {"F1": 0.0, "threshold": 0.5} for label in label_list} for lang in langs
    }

    for base_threshold in thresholds:
        adjusted_preds = np.zeros_like(probs)

        for i in range(probs.shape[0]):
            sentiment_class = sentiment_label_map[sentiment_preds[i]]
            thresholds_i = np.ones(len(label_list)) * base_threshold

            if sentiment_class == "Negative":
                thresholds_i[1:] -= 0.1
                thresholds_i[1:] = np.clip(thresholds_i[1:], 0.1, 0.5)
            elif sentiment_class == "Neutral":
                thresholds_i[0] -= 0.1
                thresholds_i[1:] += 0.1
                thresholds_i = np.clip(thresholds_i, 0.1, 0.9)
            elif sentiment_class == "Positive":
                thresholds_i += 0.1
                thresholds_i = np.clip(thresholds_i, 0.5, 0.9)

            adjusted_preds[i] = (probs[i] >= thresholds_i).astype(int)

        # === Global Best Per-Class Threshold ===
        per_class_f1 = []
        for j, label in enumerate(label_list):
            tp = ((adjusted_preds[:, j] == 1) & (y_true[:, j] == 1)).sum()
            pred_count = adjusted_preds[:, j].sum()
            gold_count = y_true[:, j].sum()
            f1 = f1_pyevall_style(tp, pred_count, gold_count)
            per_class_f1.append(f1)

            if f1 > best_per_class[label]["F1"]:
                best_per_class[label]["F1"] = f1
                best_per_class[label]["threshold"] = float(base_threshold)

        macro_f1 = np.mean(per_class_f1)
        macro_f1_per_threshold[float(base_threshold)] = macro_f1

        # === Per-language Per-class Thresholds ===
        for lang in langs:
            indices = np.where(lang_ids == lang)[0]
            if len(indices) == 0:
                continue

            lang_preds = adjusted_preds[indices]
            lang_true = y_true[indices]

            for j, label in enumerate(label_list):
                tp = ((lang_preds[:, j] == 1) & (lang_true[:, j] == 1)).sum()
                pred_count = lang_preds[:, j].sum()
                gold_count = lang_true[:, j].sum()
                f1 = f1_pyevall_style(tp, pred_count, gold_count)

                if f1 > best_thresh_per_lang_class[lang][label]["F1"]:
                    best_thresh_per_lang_class[lang][label]["F1"] = f1
                    best_thresh_per_lang_class[lang][label]["threshold"] = float(base_threshold)

    # === Final Results ===
    best_overall_threshold = max(macro_f1_per_threshold, key=macro_f1_per_threshold.get)
    best_macro_f1 = macro_f1_per_threshold[best_overall_threshold]

    results = {
        "per_class": best_per_class,
        "best_overall_threshold": best_overall_threshold,
        "best_overall_FMeasure": round(best_macro_f1, 4),
        "best_thresholds_per_language_class": best_thresh_per_lang_class,
    }

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    return results, best_overall_threshold

 
# Function to calculate and save multi-label classification metrics 
def calculate_metrics(y_true, y_pred, class_names, save_directory_name1, save_directory_name2, save_directory_name3, thresh, sigmoid=True):
    """
    Calculate and save multi-label classification metrics using a fixed threshold.

    Args:
        y_true (np.ndarray): True labels (binary matrix).
        y_pred (np.ndarray): Model predictions (logits).
        class_names (list): List of label names.
        save_directory_name1 (str): Path to save classification report.
        save_directory_name2 (str): Path to save confusion matrices.
        save_directory_name3 (str): Path to save general confusion matrix.
        thresh (float): Best threshold applied to all labels.
        sigmoid (bool): Apply sigmoid activation to predictions.

    Returns:
        pd.DataFrame: DataFrame containing F1, precision, and recall per class.
    """
    print('\nCALCULATING METRICS...')

    assert len(y_pred) == len(y_true)

    # Convert predictions to probabilities if needed
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)

    if sigmoid:
        y_pred = y_pred.sigmoid()

    # Apply a single threshold across all labels
    y_pred_thresh = (y_pred >= thresh).numpy()
    y_true = y_true.bool().numpy()

    # Compute per-class scores
    scores = {}
    for i, label in enumerate(class_names):
        scores[label] = {
            'f1': round(f1_score(y_true[:, i], y_pred_thresh[:, i], zero_division=0), 2),
            'precision': round(precision_score(y_true[:, i], y_pred_thresh[:, i], zero_division=0), 2),
            'recall': round(recall_score(y_true[:, i], y_pred_thresh[:, i], zero_division=0), 2)
        }

    # Compute macro-averaged scores
    macro_f1 = f1_score(y_true, y_pred_thresh, average='macro', zero_division=0)
    macro_precision = precision_score(y_true, y_pred_thresh, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred_thresh, average='macro', zero_division=0)

    scores['macro'] = {
        'f1': round(macro_f1, 2),
        'precision': round(macro_precision, 2),
        'recall': round(macro_recall, 2)
    }

    # print(f"Scores: {scores}")

    # Save classification report
    report = classification_report(y_true, y_pred_thresh, target_names=class_names, zero_division=0, digits=4)
    with open(save_directory_name1, 'w') as f:
        f.write(report)

    # === Save Per-Class Confusion Matrices ===
    figsize = max(10, len(class_names))
    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred_thresh)
    # print("Confusion Matrices:")
    for i, matrix in enumerate(confusion_matrices):
        # print(f"Label {class_names[i]}:")
        # print(matrix)
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        sns.heatmap(matrix, annot=True, fmt='d', ax=ax, cmap="Blues")
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title(f'Confusion Matrix for {class_names[i]}')
        plt.savefig(f'{save_directory_name2}_{class_names[i]}_matrix.png', bbox_inches="tight")
        plt.close()

    # === Save General Confusion Matrix ===
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred_thresh.argmax(axis=1))
    unique_labels = np.unique(np.concatenate((y_true.argmax(axis=1), y_pred_thresh.argmax(axis=1))))
    filtered_class_names = [class_names[i] for i in unique_labels]

    df_cm = pd.DataFrame(cm, index=filtered_class_names, columns=filtered_class_names)
    plt.figure(figsize=(figsize, figsize))
    hmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.title('General Confusion Matrix')
    plt.savefig(save_directory_name3, bbox_inches='tight')
    plt.close()

    scores_df = pd.DataFrame(scores).T
    return scores_df


# Function to calculate and save multi-label classification metrics using per-class thresholds
def calculate_metrics_perclass_threshs(
    y_true,
    y_pred,
    class_names,
    save_directory_name1,
    save_directory_name2,
    save_directory_name3,
    best_thresholds_f1_scores,
    sigmoid=True
):
    """
    Calculate and save multi-label classification metrics using per-class thresholds from find_best_thresholds.

    Args:
        y_true (np.ndarray): True binary labels.
        y_pred (np.ndarray): Raw model predictions (logits).
        class_names (list): List of all class names.
        save_directory_name1 (str): File path to save classification report.
        best_thresholds_f1_scores (dict): Dictionary from find_best_thresholds with "per_class" thresholds.
        sigmoid (bool): Whether to apply sigmoid to logits.

    Returns:
        pd.DataFrame: DataFrame with precision, recall, F1 for each class and overall metrics.
    """
    print('\nCALCULATING METRICS PER CLASS...')
    assert len(y_pred) == len(y_true)

    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)

    if sigmoid:
        y_pred = y_pred.sigmoid()

    y_true = y_true.bool().numpy()
    y_pred_thresh = y_pred.clone()

    # Apply best per-class thresholds
    for i, label in enumerate(class_names):
        threshold = best_thresholds_f1_scores.get("per_class", {}).get(label, {}).get("threshold", 0.5)
        y_pred_thresh[:, i] = (y_pred[:, i] >= threshold)

    y_pred_thresh = y_pred_thresh.numpy()

    # Compute per-class metrics
    scores = {}
    for i, label in enumerate(class_names):
        scores[label] = {
            'f1': round(f1_score(y_true[:, i], y_pred_thresh[:, i], zero_division=0), 2),
            'precision': round(precision_score(y_true[:, i], y_pred_thresh[:, i], zero_division=0), 2),
            'recall': round(recall_score(y_true[:, i], y_pred_thresh[:, i], zero_division=0), 2)
        }

    # Compute macro-averaged metrics
    macro_f1 = f1_score(y_true, y_pred_thresh, average='macro', zero_division=0)
    macro_precision = precision_score(y_true, y_pred_thresh, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred_thresh, average='macro', zero_division=0)

    # Mean F1 (excluding "NO")
    idx_no = class_names.index("NO") if "NO" in class_names else -1
    f1_values = [scores[label]["f1"] for i, label in enumerate(class_names) if i != idx_no]
    mean_f1 = np.mean(f1_values)

    # Store macro and mean scores
    scores['macro'] = {
        'f1': round(macro_f1, 2),
        'precision': round(macro_precision, 2),
        'recall': round(macro_recall, 2)
    }
    scores['mean_f1'] = round(mean_f1, 2)
    scores['best_threshold_NO'] = best_thresholds_f1_scores.get("per_class", {}).get("NO", {}).get("threshold", 0.5)

    # print(f"Scores: {scores}")

    # Save classification report
    report = classification_report(y_true, y_pred_thresh, target_names=class_names, zero_division=0, digits=4)
    with open(save_directory_name1, 'w') as f:
        f.write(report)

    # === Save Per-Class Confusion Matrices ===
    figsize = max(10, len(class_names))
    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred_thresh)
    # print("Confusion Matrices:")
    for i, matrix in enumerate(confusion_matrices):
        # print(f"Label {class_names[i]}:")
        # print(matrix)
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        sns.heatmap(matrix, annot=True, fmt='d', ax=ax, cmap="Blues")
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title(f'Confusion Matrix for {class_names[i]}')
        plt.savefig(f'{save_directory_name2}_{class_names[i]}_matrix.png', bbox_inches="tight")
        plt.close()

    # === Save General Confusion Matrix ===
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred_thresh.argmax(axis=1))
    unique_labels = np.unique(np.concatenate((y_true.argmax(axis=1), y_pred_thresh.argmax(axis=1))))
    filtered_class_names = [class_names[i] for i in unique_labels]

    df_cm = pd.DataFrame(cm, index=filtered_class_names, columns=filtered_class_names)
    plt.figure(figsize=(figsize, figsize))
    hmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.title('General Confusion Matrix')
    plt.savefig(save_directory_name3, bbox_inches='tight')
    plt.close()

    return pd.DataFrame(scores).T


# Function to calculate and save multi-task classification metrics
def calculate_metrics_multitask(
    y_true_sexist,
    y_pred_sexist,
    y_true_sentiment,
    y_pred_sentiment,
    class_names,
    sentiment_labels,
    save_directory_name1,
    save_directory_name2,
    save_directory_name3,
    save_directory_name4,
    thresh,
    sigmoid=True
):
    """
    Calculate and save metrics for multi-label (sexism) and multi-class (sentiment) classification.

    Args:
        y_true_sexist (np.ndarray): True binary labels for sexism.
        y_pred_sexist (np.ndarray): Raw logits for sexism predictions.
        y_true_sentiment (np.ndarray): True one-hot encoded sentiment labels.
        y_pred_sentiment (np.ndarray): Raw logits for sentiment predictions.
        class_names (list): List of sexism label names.
        sentiment_labels (list): List of sentiment class names.
        save_directory_name1 (str): File path to save the sexism classification report.
        save_directory_name2 (str): Prefix path for individual sexism confusion matrices.
        save_directory_name3 (str): Path to save general sexism confusion matrix.
        save_directory_name4 (str): File path to save the sentiment classification report.
        thresh (float): Threshold for binarizing sexism predictions.
        sigmoid (bool): Whether to apply sigmoid to predictions.

    Returns:
        pd.DataFrame: Sexism classification scores per class + macro F1/precision/recall.
    """
    print('\nCALCULATING MULTITASK METRICS...')

    # ==== SEXISM (multi-label) ====
    assert len(y_pred_sexist) == len(y_true_sexist)
    y_pred_sexist = torch.from_numpy(y_pred_sexist)
    y_true_sexist = torch.from_numpy(y_true_sexist)

    if sigmoid:
        y_pred_sexist = y_pred_sexist.sigmoid()

    y_pred_thresh = (y_pred_sexist >= thresh).numpy()
    y_true_sexist = y_true_sexist.bool().numpy()

    scores = {}
    for i, label in enumerate(class_names):
        scores[label] = {
            'f1': round(f1_score(y_true_sexist[:, i], y_pred_thresh[:, i], zero_division=0), 2),
            'precision': round(precision_score(y_true_sexist[:, i], y_pred_thresh[:, i], zero_division=0), 2),
            'recall': round(recall_score(y_true_sexist[:, i], y_pred_thresh[:, i], zero_division=0), 2)
        }

    macro_f1 = f1_score(y_true_sexist, y_pred_thresh, average='macro', zero_division=0)
    macro_precision = precision_score(y_true_sexist, y_pred_thresh, average='macro', zero_division=0)
    macro_recall = recall_score(y_true_sexist, y_pred_thresh, average='macro', zero_division=0)

    scores['macro'] = {
        'f1': round(macro_f1, 2),
        'precision': round(macro_precision, 2),
        'recall': round(macro_recall, 2)
    }

    # Save sexism classification report
    report = classification_report(y_true_sexist, y_pred_thresh, target_names=class_names, zero_division=0, digits=4)
    with open(save_directory_name1, 'w') as f:
        f.write(report)

    # Per-class confusion matrices (sexism)
    figsize = max(10, len(class_names))
    confusion_matrices = multilabel_confusion_matrix(y_true_sexist, y_pred_thresh)
    for i, matrix in enumerate(confusion_matrices):
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        sns.heatmap(matrix, annot=True, fmt='d', ax=ax, cmap="Blues")
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title(f'Confusion Matrix for {class_names[i]}')
        plt.savefig(f'{save_directory_name2}_{class_names[i]}_matrix.png', bbox_inches="tight")
        plt.close()

    # General confusion matrix (sexism)
    cm = confusion_matrix(y_true_sexist.argmax(axis=1), y_pred_thresh.argmax(axis=1))
    unique_labels = np.unique(np.concatenate((y_true_sexist.argmax(axis=1), y_pred_thresh.argmax(axis=1))))
    filtered_class_names = [class_names[i] for i in unique_labels]

    df_cm = pd.DataFrame(cm, index=filtered_class_names, columns=filtered_class_names)
    plt.figure(figsize=(figsize, figsize))
    hmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.title('General Confusion Matrix (Sexism)')
    plt.savefig(save_directory_name3, bbox_inches='tight')
    plt.close()

    # ==== SENTIMENT (multi-class) ====
    sentiment_preds = np.argmax(y_pred_sentiment, axis=1)
    sentiment_true = y_true_sentiment

    sentiment_report = classification_report(sentiment_true, sentiment_preds, target_names=sentiment_labels, digits=4)
    with open(save_directory_name4, 'w') as f:
        f.write(sentiment_report)

    return pd.DataFrame(scores).T


# Function to calculate and save multi-task classification metrics using per-class thresholds
def calculate_metrics_perclass_threshs_multitask(
    y_true_sexist,
    y_pred_sexist,
    y_true_sentiment,
    y_pred_sentiment,
    class_names,
    sentiment_labels,
    save_directory_name1,
    save_directory_name2,
    save_directory_name3,
    save_directory_name4,
    best_thresholds_f1_scores,
    languages=None,
    sigmoid=True
):
    """
    Multi-task metrics: Per-class thresholds for sexism + sentiment classification report.

    Args:
        y_true_sexist (np.ndarray): True binary labels for sexism.
        y_pred_sexist (np.ndarray): Logits for sexism predictions.
        y_true_sentiment (np.ndarray): One-hot sentiment labels.
        y_pred_sentiment (np.ndarray): Logits for sentiment predictions.
        class_names (list): List of sexism label names.
        sentiment_labels (list): List of sentiment class names.
        save_directory_name1 (str): Save path for sexism classification report.
        save_directory_name2 (str): Prefix for individual confusion matrices (sexism).
        save_directory_name3 (str): Save path for general confusion matrix (sexism).
        save_directory_name4 (str): Save path for sentiment classification report.
        best_thresholds_f1_scores (dict): Dict with per-class or per-lang+class thresholds.
        languages (list): List of language codes for each sample.
        sigmoid (bool): Whether to apply sigmoid to sexism logits.

    Returns:
        pd.DataFrame: Sexism classification scores per class and macro.
    """
    print('\nCALCULATING METRICS PER CLASS (MULTITASK)...')
    assert len(y_pred_sexist) == len(y_true_sexist)

    y_pred = torch.from_numpy(y_pred_sexist)
    y_true = torch.from_numpy(y_true_sexist)

    if sigmoid:
        y_pred = y_pred.sigmoid()

    y_true = y_true.bool().numpy()
    y_pred_thresh = y_pred.clone()

    for i in range(y_pred.shape[0]):
        lang = languages[i] if languages is not None else "default"
        for j, label in enumerate(class_names):
            threshold = (
            best_thresholds_f1_scores.get("best_thresholds_per_language_class", {}).get(lang, {}).get(label, {}).get("threshold", 0.5))
            if threshold is None:
                threshold = best_thresholds_f1_scores.get("per_class", {}).get(label, {}).get("threshold", 0.5)
            y_pred_thresh[i, j] = y_pred[i, j] >= threshold

    y_pred_thresh = y_pred_thresh.numpy()

    # === Metrics: per class
    scores = {}
    for i, label in enumerate(class_names):
        scores[label] = {
            'f1': round(f1_score(y_true[:, i], y_pred_thresh[:, i], zero_division=0), 2),
            'precision': round(precision_score(y_true[:, i], y_pred_thresh[:, i], zero_division=0), 2),
            'recall': round(recall_score(y_true[:, i], y_pred_thresh[:, i], zero_division=0), 2)
        }

    macro_f1 = f1_score(y_true, y_pred_thresh, average='macro', zero_division=0)
    macro_precision = precision_score(y_true, y_pred_thresh, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred_thresh, average='macro', zero_division=0)

    idx_no = class_names.index("NO") if "NO" in class_names else -1
    f1_values = [scores[label]["f1"] for i, label in enumerate(class_names) if i != idx_no]
    mean_f1 = np.mean(f1_values)

    scores['macro'] = {
        'f1': round(macro_f1, 2),
        'precision': round(macro_precision, 2),
        'recall': round(macro_recall, 2)
    }
    scores['mean_f1'] = round(mean_f1, 2)
    scores['best_threshold_NO'] = best_thresholds_f1_scores.get("per_class", {}).get("NO", {}).get("threshold", 0.5)

    # === Save sexism classification report
    report = classification_report(y_true, y_pred_thresh, target_names=class_names, zero_division=0, digits=4)
    with open(save_directory_name1, 'w') as f:
        f.write(report)

    # === Save per-class confusion matrices (sexism)
    figsize = max(10, len(class_names))
    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred_thresh)
    for i, matrix in enumerate(confusion_matrices):
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        sns.heatmap(matrix, annot=True, fmt='d', ax=ax, cmap="Blues")
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title(f'Confusion Matrix for {class_names[i]}')
        plt.savefig(f'{save_directory_name2}_{class_names[i]}_matrix.png', bbox_inches="tight")
        plt.close()

    # === Save general confusion matrix (sexism)
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred_thresh.argmax(axis=1))
    unique_labels = np.unique(np.concatenate((y_true.argmax(axis=1), y_pred_thresh.argmax(axis=1))))
    filtered_class_names = [class_names[i] for i in unique_labels]

    df_cm = pd.DataFrame(cm, index=filtered_class_names, columns=filtered_class_names)
    plt.figure(figsize=(figsize, figsize))
    hmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.title('General Confusion Matrix (Sexism)')
    plt.savefig(save_directory_name3, bbox_inches='tight')
    plt.close()

    # === SENTIMENT ===
    sentiment_preds = np.argmax(y_pred_sentiment, axis=1)
    sentiment_true = y_true_sentiment

    sentiment_report = classification_report(sentiment_true, sentiment_preds, target_names=sentiment_labels, digits=4)
    with open(save_directory_name4, 'w') as f:
        f.write(sentiment_report)

    return pd.DataFrame(scores).T


# Function to perform error analysis
def perform_error_analysis(val_labels, val_dataframe, prediction_json_path, label_list):
    """
    Perform error analysis using ground truth labels and predictions loaded from JSON.

    Args:
        val_labels (np.ndarray): Ground truth binary labels (from Trainer).
        val_dataframe (pd.DataFrame): DataFrame containing the validation set with 'id' column.
        prediction_json_path (str): Path to the val_predictions.json file.
        label_list (List[str]): List of label names.

    Returns:
        results_threshold, no_pred_true, no_true_pred
    """

    print("\nPERFORMING ERROR ANALYSIS...")

    # Load predictions from JSON
    with open(prediction_json_path, "r") as f:
        pred_data = json.load(f)

    # Map ID to predicted label list
    id_to_pred_labels = {entry["id"]: entry["value"] for entry in pred_data}

    label_counts_dataset = Counter()
    no_pred_but_true_labels = []
    no_true_but_pred_labels = []
    confused_sentences_labels = []
    incorrect_counts = Counter()

    id_list = val_dataframe["id"].astype(str).tolist()

    for idx, sentence_id in enumerate(id_list):
        true_labels_bin = val_labels[idx]
        true_labels = [label_list[i] for i, val in enumerate(true_labels_bin) if val == 1]
        pred_labels = id_to_pred_labels.get(sentence_id, [])

        # Count true labels for overall stats
        for label in true_labels:
            label_counts_dataset[label] += 1

        true_set = set(true_labels)
        pred_set = set(pred_labels)

        if not pred_set and true_set:
            no_pred_but_true_labels.append({
                "Sentence-ID": sentence_id,
                "Predicted_labels": [],
                "True_labels": list(true_set)
            })

        if pred_set and not true_set:
            no_true_but_pred_labels.append({
                "Sentence-ID": sentence_id,
                "Predicted_labels": list(pred_set),
                "True_labels": []
            })

        if true_set != pred_set:
            confused_sentences_labels.append({
                "Sentence-ID": sentence_id,
                "Predicted_labels": list(pred_set),
                "True_labels": list(true_set)
            })

            for label in true_set.symmetric_difference(pred_set):
                incorrect_counts[label] += 1

    total_sentences = len(val_labels)
    most_confused_labels = [label for label, count in incorrect_counts.items() if count == max(incorrect_counts.values(), default=0)]
    least_confused_labels = [label for label, count in incorrect_counts.items() if count == min(incorrect_counts.values(), default=0)]

    results_threshold = {
        "Total Sentences": total_sentences,
        "Number of no_pred_but_true_labels": len(no_pred_but_true_labels),
        "Number of no_true_but_pred_labels": len(no_true_but_pred_labels),
        "Number of confusing sentences": len(confused_sentences_labels),
        "Percentage of no_pred_but_true_labels": round(len(no_pred_but_true_labels) / total_sentences * 100, 2),
        "Percentage of no_true_but_pred_labels": round(len(no_true_but_pred_labels) / total_sentences * 100, 2),
        "Percentage of confused_sentences_labels": round(len(confused_sentences_labels) / total_sentences * 100, 2),
        "Most Confused Labels": most_confused_labels,
        "Least Confused Labels": least_confused_labels,
        "Incorrect Predicted Label Counts": dict(incorrect_counts),
        "True Label Counts": dict(label_counts_dataset),
        "Confused_sentences_labels": confused_sentences_labels
    }

    no_pred_true = {"No_predictions_but_true_labels": no_pred_but_true_labels}
    no_true_pred = {"No_true_labels_but_predictions": no_true_but_pred_labels}

    return results_threshold, no_pred_true, no_true_pred


# Function to perform error analysis for multitask models
def perform_error_analysis_multitask(
    val_labels_sexist,
    val_labels_sentiment,
    val_dataframe,
    prediction_json_path,
    label_list,
    sentiment_labels,
):
    """
    Perform multitask error analysis using ground truth labels and predictions loaded from JSON.

    Returns:
        - error_stats (dict): Overall metrics and counts.
        - no_pred_but_true_labels (list): Cases with no predicted labels but ground truth present.
        - no_true_but_pred_labels (list): Cases with predicted labels but no ground truth.
    """
    print("\n[INFO] Performing multitask error analysis...")

    with open(prediction_json_path, "r") as f:
        pred_data = json.load(f)

    id_to_pred_labels = {entry["id"]: entry["value"] for entry in pred_data}
    id_to_pred_sentiment = {entry["id"]: entry.get("sentiment_prediction", None) for entry in pred_data}

    label_counts_dataset = Counter()
    no_pred_but_true_labels = []
    no_true_but_pred_labels = []
    confused_sexist_labels = []
    incorrect_counts = Counter()
    sentiment_errors = []

    id_list = val_dataframe["id"].astype(str).tolist()

    for idx, sentence_id in enumerate(id_list):
        # === TRUE LABELS ===
        true_labels_bin = val_labels_sexist[idx]
        true_labels = [label_list[i] for i, val in enumerate(true_labels_bin) if val == 1]

        true_sentiment_idx = val_labels_sentiment[idx]
        if isinstance(true_sentiment_idx, (np.ndarray, list)):
            true_sentiment_idx = int(np.argmax(true_sentiment_idx))
        else:
            true_sentiment_idx = int(true_sentiment_idx)
        true_sentiment = sentiment_labels[true_sentiment_idx]

        # === PREDICTED LABELS ===
        pred_labels = id_to_pred_labels.get(sentence_id, [])
        pred_set = set(pred_labels if isinstance(pred_labels, list) else [pred_labels])
        pred_sentiment = id_to_pred_sentiment.get(sentence_id)

        true_set = set(true_labels)
        for label in true_labels:
            label_counts_dataset[label] += 1

        if not pred_set and true_set:
            no_pred_but_true_labels.append({
                "Sentence-ID": sentence_id,
                "Predicted_labels": [],
                "True_labels": list(true_set)
            })

        if pred_set and not true_set:
            no_true_but_pred_labels.append({
                "Sentence-ID": sentence_id,
                "Predicted_labels": list(pred_set),
                "True_labels": []
            })

        if true_set != pred_set:
            confused_sexist_labels.append({
                "Sentence-ID": sentence_id,
                "Predicted_labels": list(pred_set),
                "True_labels": list(true_set)
            })
            for label in true_set.symmetric_difference(pred_set):
                incorrect_counts[label] += 1

        # === SENTIMENT ERROR ===
        if pred_sentiment != true_sentiment:
            sentiment_errors.append({
                "Sentence-ID": sentence_id,
                "Predicted_sentiment": pred_sentiment,
                "True_sentiment": true_sentiment
            })

    total_sentences = len(val_labels_sexist)
    most_confused = [label for label, count in incorrect_counts.items() if count == max(incorrect_counts.values(), default=0)]
    least_confused = [label for label, count in incorrect_counts.items() if count == min(incorrect_counts.values(), default=0)]

    error_stats = {
        "Total Sentences": total_sentences,
        "Sexism - No prediction but true labels": len(no_pred_but_true_labels),
        "Sexism - No true label but predicted": len(no_true_but_pred_labels),
        "Sexism - Confused predictions": len(confused_sexist_labels),
        "Sentiment - Incorrect predictions": len(sentiment_errors),
        "Sexism - % No prediction but true": round(len(no_pred_but_true_labels) / total_sentences * 100, 2),
        "Sexism - % No true label but predicted": round(len(no_true_but_pred_labels) / total_sentences * 100, 2),
        "Sexism - % Confused": round(len(confused_sexist_labels) / total_sentences * 100, 2),
        "Sentiment - % Incorrect": round(len(sentiment_errors) / total_sentences * 100, 2),
        "Most Confused Sexism Labels": most_confused,
        "Least Confused Sexism Labels": least_confused,
        "Incorrect Predicted Label Counts": dict(incorrect_counts),
        "True Label Counts": dict(label_counts_dataset),
        "Confused Sentences (Sexism)": confused_sexist_labels,
        "Sentiment Errors": sentiment_errors,
    }

    return error_stats, no_pred_but_true_labels, no_true_but_pred_labels


# Function to train, evaluate, and predict using the Trainer
def train_evaluate_predict(
    trainer,
    model,
    tokenizer,
    output_dir,
    encoded_train_dataset,
    encoded_dev_dataset,
    encoded_test_dataset=None,
    multitask=False,
    label_list=None,
    sentiment_labels=None,
    thresholds=np.arange(0.1, 1.0, 0.05),
):
    print("TRAINING MODEL...")
    train_result = trainer.train(resume_from_checkpoint=False)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = train_result.metrics
    metrics["train_samples"] = len(encoded_train_dataset)
    trainer.log_metrics("train", flatten_dict(metrics))
    trainer.save_metrics("train", flatten_dict(metrics))
    trainer.save_state()

    print("EVALUATING MODEL...")
    eval_metrics = trainer.evaluate(encoded_dev_dataset)
    eval_metrics["eval_samples"] = len(encoded_dev_dataset)
    trainer.log_metrics("eval", flatten_dict(eval_metrics))
    trainer.save_metrics("eval", flatten_dict(eval_metrics))

    print("PREDICTING LABELLED DATA...")
    preds = trainer.predict(encoded_dev_dataset)

    if multitask:
        # Expected from SequenceClassifierOutput: tuple of (logits_sexist, logits_sentiment)
        logits_sexist, logits_sentiment, language_ids = preds.predictions
        labels_sexist, labels_sentiment = preds.label_ids
    else:
        logits_sexist = preds.predictions
        labels_sexist = preds.label_ids
        logits_sentiment = None
        labels_sentiment = None
        language_ids = None

    metrics = preds.metrics
    metrics["predict_samples"] = len(encoded_dev_dataset)
    trainer.log_metrics("predict", flatten_dict(metrics))
    trainer.save_metrics("predict", flatten_dict(metrics))

    # === PREDICT ON TEST ===
    if encoded_test_dataset is not None:
        print("\nPREDICTING TEST DATA...")
        test_preds = trainer.predict(encoded_test_dataset)
        if multitask:
            test_logits_sexist, test_logits_sentiment, test_language_ids = test_preds.predictions
            return logits_sexist, logits_sentiment, labels_sexist, labels_sentiment, language_ids, test_logits_sexist, test_logits_sentiment, test_language_ids
        else:
            return logits_sexist, labels_sexist, test_preds.predictions

    return logits_sexist, logits_sentiment, labels_sexist, labels_sentiment, language_ids


# Function to generate predictions file
def generate_predictions_file(trainer, dataset, dataframe, output_json, categories, metrics, evaluation_type="soft", id_column="id"):
    """
    Generate a JSON file with model predictions using the best threshold for each label.

    Supports:
    - Hard labels (binary predictions using per-label thresholds).
    - Soft labels (probability scores for PyEval soft evaluation).

    Args:
        trainer (Trainer): Hugging Face Trainer object.
        dataset (Dataset): Evaluation dataset.
        dataframe (pd.DataFrame): Dataframe containing original IDs for each instance.
        output_json (str): Path to save the predictions file.
        categories (list): List of label names (including "NO" and sexism categories).
        metrics (dict): Dictionary returned by `find_best_thresholds`, containing per-label best thresholds.
        id_column (str): Column name in `dataframe` that contains the unique ID for each instance.
        evaluation_type (str): Type of evaluation ("soft" or "hard").

    Returns:
        str: Path to the saved predictions file.
    """
    print("\nGenerating predictions for PyEval...")

    # Check if the required keys are present in the metrics dictionary
    if "per_class" not in metrics:
        raise KeyError("The metrics dictionary does not contain the required key 'per_class'.")

    # Extract best per-label thresholds from `metrics`
    best_thresholds_dict = {k: v["threshold"] for k, v in metrics["per_class"].items()}

    # Run model prediction
    predictions_output = trainer.predict(dataset)
    logits = predictions_output.predictions  # Model raw logits

    # Convert logits to probabilities using sigmoid
    probabilities = expit(logits)

    predictions_list = []

    # Get actual IDs from the dataframe
    ids = dataframe[id_column].tolist()  # Extract actual IDs from the dataframe

    for idx, prob in enumerate(probabilities):
        instance_id = str(ids[idx])  # Get the actual ID from `dataframe`
        
        if evaluation_type == "soft":
            # SOFT LABELS: Store probability values
            prediction_dict = {
                "test_case": "EXIST2025",
                "id": instance_id,
                "value": {categories[i]: float(prob[i]) for i in range(len(categories))}
            }
        else:
            # HARD LABELS: Apply threshold per label and store binary predictions
            pred_labels = [categories[i] for i in range(len(categories)) if prob[i] >= best_thresholds_dict.get(categories[i], 0.5)]

            # If no category was predicted, assign "NO"
            if not pred_labels:
                pred_labels = ["NO"]

            prediction_dict = {
                "test_case": "EXIST2025",
                "id": instance_id,
                "value": pred_labels
            }

        predictions_list.append(prediction_dict)

    # Save predictions to JSON
    with open(output_json, "w", encoding='utf-8') as f:
        json.dump(predictions_list, f, indent=2)

    print(f"Saved predictions to {output_json}")

    return output_json


def generate_predictions_file_sexism(
    logits_sexist,
    dataframe,
    output_json,
    categories,
    metrics,
    evaluation_type="soft",
    id_column="id"
):
    """
    Generate a JSON file with sexism predictions only.

    Args:
        logits_sexist (np.ndarray): Raw model logits for the sexism task.
        dataframe (pd.DataFrame): Original dataframe with IDs.
        output_json (str): Path to save predictions file.
        categories (list): List of sexism label names.
        metrics (dict): Output from `find_best_thresholds`, containing per-class thresholds.
        evaluation_type (str): "soft" or "hard".
        id_column (str): Column in dataframe containing sentence IDs.
    """
    print("\nGenerating predictions...")

    # Check thresholds exist
    if "per_class" not in metrics:
        raise KeyError("The metrics dictionary does not contain 'per_class' thresholds.")

    best_thresholds = {label: v["threshold"] for label, v in metrics["per_class"].items()}

    # Convert logits to probabilities
    probabilities = expit(logits_sexist)  # Apply sigmoid

    predictions_list = []
    ids = dataframe[id_column].astype(str).tolist()

    for idx, prob in enumerate(probabilities):
        instance_id = ids[idx]

        if evaluation_type == "soft":
            prediction_dict = {
                "test_case": "EXIST2025",
                "id": instance_id,
                "value": {categories[i]: float(prob[i]) for i in range(len(categories))}
            }
        else:
            pred_labels = [categories[i] for i, p in enumerate(prob) if p >= best_thresholds.get(categories[i], 0.5)]

            if not pred_labels:
                pred_labels = ["NO"]

            prediction_dict = {
                "test_case": "EXIST2025",
                "id": instance_id,
                "value": pred_labels
            }

        predictions_list.append(prediction_dict)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(predictions_list, f, indent=2)

    return output_json


# Function to generate predictions file for sexism task with per-language thresholds
def generate_predictions_file_sexism_multitask(
    logits_sexist,
    dataframe,
    output_json,
    categories,
    metrics,
    evaluation_type="soft",
    id_column="id",
    lang_column="lang"
):
    """
    Generate a JSON file with sexism predictions using per-language, per-class thresholds.

    Args:
        logits_sexist (np.ndarray): Raw model logits for the sexism task.
        dataframe (pd.DataFrame): Original dataframe with IDs and languages.
        output_json (str): Path to save predictions file.
        categories (list): List of sexism label names.
        metrics (dict): Output from `find_best_thresholds_multitask`.
        evaluation_type (str): "soft" or "hard".
        id_column (str): Column in dataframe containing sentence IDs.
        lang_column (str): Column in dataframe containing language ("en", "es").
    """
    print("\nGenerating predictions (SEXISM only...")

    if "per_class" not in metrics or "best_thresholds_per_language_class" not in metrics:
        raise KeyError("Missing required keys in metrics for per-language thresholding.")

    # Fallback (general per-class thresholds)
    general_thresholds = {label: v["threshold"] for label, v in metrics["per_class"].items()}
    lang_class_thresholds = metrics["best_thresholds_per_language_class"]

    probabilities = expit(logits_sexist)
    predictions_list = []

    ids = dataframe[id_column].astype(str).tolist()
    langs = dataframe[lang_column].astype(str).tolist()

    for idx, prob in enumerate(probabilities):
        instance_id = ids[idx]
        lang = langs[idx]

        if evaluation_type == "soft":
            prediction_dict = {
                "test_case": "EXIST2025",
                "id": instance_id,
                "value": {categories[i]: float(prob[i]) for i in range(len(categories))}
            }
        else:
            pred_labels = []
            for i, p in enumerate(prob):
                label = categories[i]
                threshold = lang_class_thresholds.get(lang, {}).get(label, {}).get("threshold", general_thresholds[label])
                if p >= threshold:
                    pred_labels.append(label)

            if not pred_labels:
                pred_labels = ["NO"]

            prediction_dict = {
                "test_case": "EXIST2025",
                "id": instance_id,
                "value": pred_labels
            }

        predictions_list.append(prediction_dict)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(predictions_list, f, indent=2)

    return output_json


# Function to generate sexism and sentiment predictions file for multitask model with per-language thresholds
def generate_multitask_predictions_file(
    logits_sexist,
    logits_sentiment,
    dataframe,
    output_json,
    sexism_labels,
    metrics,
    evaluation_type="soft",
    id_column="id",
    lang_column="lang",
    sentiment_label_names=("Positive", "Neutral", "Negative")
):
    """
    Generate predictions JSON file for multi-task model (sexism + sentiment) using
    per-language and per-class thresholds for sexism classification.

    Args:
        logits_sexist (np.ndarray): Raw logits from the sexism head.
        logits_sentiment (np.ndarray): Raw logits from the sentiment head.
        dataframe (pd.DataFrame): DataFrame containing example IDs and language.
        output_json (str): Path to save the output JSON file.
        sexism_labels (list): List of sexism category labels (including "NO").
        metrics (dict): Output from `find_best_thresholds_multitask`.
        evaluation_type (str): "soft" or "hard".
        id_column (str): Column in dataframe containing example IDs.
        lang_column (str): Column in dataframe containing language.
        sentiment_label_names (tuple): Ordered sentiment class names.

    Returns:
        str: Path to the saved prediction JSON.
    """
    print("\n[INFO] Generating multitask predictions (SEXISM + SENTIMENT)...")

    # Load general and per-language thresholds
    general_thresholds = {label: v["threshold"] for label, v in metrics.get("per_class", {}).items()}
    lang_class_thresholds = metrics.get("best_thresholds_per_language_class", {})

    # Convert logits to probabilities
    probs_sexist = expit(logits_sexist)
    probs_sentiment = torch.softmax(torch.tensor(logits_sentiment), dim=1).numpy()

    predictions_list = []
    ids = dataframe[id_column].astype(str).tolist()
    langs = dataframe[lang_column].astype(str).tolist()

    for i in range(len(ids)):
        instance_id = ids[i]
        lang = langs[i]

        # === Sexism Prediction ===
        if evaluation_type == "soft":
            sexism_output = {sexism_labels[j]: float(probs_sexist[i][j]) for j in range(len(sexism_labels))}
        else:
            pred_labels = []
            for j, label in enumerate(sexism_labels):
                threshold = lang_class_thresholds.get(lang, {}).get(label, {}).get("threshold", general_thresholds[label])
                if probs_sexist[i][j] >= threshold:
                    pred_labels.append(label)

            if not pred_labels:
                pred_labels = ["NO"]
            sexism_output = pred_labels

        # === Sentiment Prediction ===
        sentiment_class_index = int(np.argmax(probs_sentiment[i]))
        sentiment_output = sentiment_label_names[sentiment_class_index]

        prediction_dict = {
            "test_case": "EXIST2025",
            "id": instance_id,
            "value": sexism_output,
            "sentiment_prediction": sentiment_output
        }

        predictions_list.append(prediction_dict)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(predictions_list, f, indent=2)

    print(f"[INFO] Saved multitask predictions to: {output_json}")
    return output_json


# Function to evaluate the model using PyEval
def evaluate_model_pyeval(predictions_json, gold_json, mode="hard", verbose=True):
    """
    Evaluate the model using PyEval based on the specified mode (hard-hard or soft-soft).

    Args:
        predictions_json (str): Path to the predictions file.
        gold_json (str): Path to the gold labels file.
        mode (str): Evaluation mode: "hard" or "soft".
        verbose (bool): Whether to print the evaluation report.

    Returns:
        dict: Dictionary containing the evaluation results.
    """
    evaluator = PyEvALLEvaluation()

    # Define the hierarchical structure for classification
    TASK_HIERARCHY = {
        "YES": ["IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE", "OBJECTIFICATION",
                "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE"],
        "NO": []
    }

    params= dict()
    params[PyEvALLUtils.PARAM_HIERARCHY]= TASK_HIERARCHY
    params[PyEvALLUtils.PARAM_REPORT]= PyEvALLUtils.PARAM_OPTION_REPORT_DATAFRAME  

    # Select metrics based on mode
    if mode == "hard":
        metrics = ["ICM", "ICMNorm", "FMeasure"]  # Hard-hard evaluation
    elif mode == "soft":
        metrics = ["ICMSoft", "ICMSoftNorm", "FMeasure"]  # Soft-soft evaluation
    else:
        raise ValueError("Invalid mode! Choose either 'hard' or 'soft'.")

    report = evaluator.evaluate(predictions_json, gold_json, metrics, **params)
  
    results = {}
    for metric in metrics:
        metric_key = metric.replace("FMeasure", "F1").replace("ICMSoft", "ICM-Soft").replace("ICMNorm", "ICM-Norm").replace("ICMSoftNorm", "ICM-Soft-Norm")
        
        metric_value = report.df_average[metric_key].iloc[0]

        try:
            results[metric_key] = float(metric_value)
        except ValueError:
            results[metric_key] = None
    
    # Retrieve per-class metric values
    classes_values = {}
    if report.df_test_case_classes is not None and not report.df_test_case_classes.empty:
        classes_values = report.df_test_case_classes.drop(axis='columns', labels='files').to_dict('records')[0]

    # Combine results into a single dictionary
    evaluation_results = {
        "overall_results": results,
        "per_class_results": classes_values
    }

    if verbose:
        report.print_report()

    return evaluation_results


# Function to load best thresholds from a checkpoint
def load_best_thresholds_from_checkpoint(checkpoint_path):
    """
    Loads the best thresholds (per-class) from a saved model checkpoint.

    Args:
        checkpoint_path (str): Path to the best model checkpoint directory.

    Returns:
        dict: Dictionary containing best thresholds and F1 scores per class.
              Returns None if thresholds file is not found.
    """
    thresholds_file = os.path.join(checkpoint_path, "best_thresholds.json")

    if not os.path.exists(thresholds_file):
        print(f"[WARNING] No best_thresholds.json found in checkpoint: {checkpoint_path}")
        return None

    with open(thresholds_file, "r") as f:
        best_thresholds = json.load(f)

    print(f"[INFO] Loaded best thresholds from: {thresholds_file}")
    return best_thresholds
    
