import os
import numpy as np
import pandas as pd
import torch
from transformers import  BitsAndBytesConfig, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, DataCollatorWithPadding
from scipy.special import expit
from datasets import Dataset
from train_arguments2 import parse_args
import numpy as np
from skimpy import skim
import csv
from collections import Counter
from peft import get_peft_model, LoraConfig, PromptTuningConfig, TaskType, PromptTuningInit, PromptLearningConfig
from typing import Literal 
import json
import random
from transformers.utils import logging
import warnings
import ast
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from transformers.trainer_utils import enable_full_determinism
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, multilabel_confusion_matrix
)
from model_architecture import MultiTask_MultiHead_XLMRoberta
from utils import (
    flatten_dict, set_seed, check_create_path, process_dataset, tokenize, tokenize_sentiment_sexism, compute_positive_weights, compute_train_number, compute_class_frequencies,
    compute_metrics, DataCollatorMultiTask, compute_positive_weights_lang, generate_predictions_file_sexism_multitask,
    MultiTaskTrainer, MultiLabelTrainer, find_best_thresholds, find_best_thresholds_multitask, calculate_metrics, calculate_metrics_perclass_threshs, calculate_metrics_multitask, calculate_metrics_perclass_threshs_multitask,
    perform_error_analysis, perform_error_analysis_multitask, generate_predictions_file_sexism,  load_best_thresholds_from_checkpoint, generate_multitask_predictions_file, compute_metrics_multitask, write_tsv_dataframe, train_evaluate_predict, generate_predictions_file, evaluate_model_pyeval
)

# Ignore warnings
warnings.filterwarnings("ignore")
logging.set_verbosity(logging.DEBUG)
logging.enable_progress_bar() 

for i in range(torch.cuda.device_count()):
    info = torch.cuda.get_device_properties(i)
    print(f"CUDA:{i} {info.name}, {info.total_memory / 1024 ** 2}MB")

# Set up GPU for Training
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device('cpu')
print(f"Using GPU is CUDA:{os.environ['CUDA_VISIBLE_DEVICES']}")

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

SENTIMENT_LABELS = ["Positive", "Neutral", "Negative"]

def main(args):

    # Sign in personal Hugging Face account
    #notebook_login()

    # Set the wandb project where this run will be logged
    #wandb.init(project = args.wandb_project, name = args.wandb_name)
    
    set_seed(args.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, 
                                              use_fast = args.use_fast,
                                              do_lower_case = args.do_lower_case, 
                                              )  


    if args.model_architecture == "baseline":
        
        data_collator = DataCollatorWithPadding(tokenizer = tokenizer, pad_to_multiple_of = 8) 
        
        if args.curriculum_learning == True:
            print("Sorting texts based on level of difficulty...")
            # Load and process data
            train_dataset, cl_train_df, train_df, train_max_len = process_dataset(args.train_dataset_path, args.evaluation_type, args.label_column, args.text_column, args.sentiment_column, args.language, tokenizer, baseline=True, sentiment=False, curriculum_learning=True)
            remove_train_columns = ["id", "tweet", "difficulty", "sentiment", "lang"]
        else:
            # Load and process data
            train_dataset, train_df, train_max_len = process_dataset(args.train_dataset_path, args.evaluation_type, args.label_column, args.text_column, args.sentiment_column, args.language, tokenizer, baseline=True, sentiment=False, curriculum_learning=False)
            remove_train_columns = ["id", "tweet", "lang"]

        test_dataset, test_df, test_max_len = process_dataset(args.test_dataset_path, args.evaluation_type, None, args.text_column, args.sentiment_column, args.language, tokenizer,  baseline=True, sentiment=False, curriculum_learning=False)  
        remove_columns = ["id", "tweet", "lang"]
        
        train_class_weights = compute_positive_weights(train_df, LABEL_LIST, label_column="labels")
        class_freq = compute_class_frequencies(train_df, LABEL_LIST, label_column="labels")
        train_num = compute_train_number(train_df)

        # Tokenize datasets
        encoded_train_dataset = train_dataset.map(
            lambda x: tokenize(x, args.text_column, tokenizer, train_max_len, args),
            batched=True,
            remove_columns=remove_train_columns,
            load_from_cache_file=False,
            desc="Running tokenizer on training dataset..."
        )

        encoded_test_dataset = test_dataset.map(
            lambda x: tokenize(x, args.text_column, tokenizer, test_max_len, args),
            batched=True,
            remove_columns=remove_columns,
            load_from_cache_file=False,
            desc="Running tokenizer on test dataset..."
        )

        columns = ["input_ids", "attention_mask", "labels"]
        encoded_train_dataset.set_format(type="torch", columns=columns)
        encoded_test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        print(f"Train dataset: {len(encoded_train_dataset)} tweets")
        print(f"Test dataset: {len(encoded_test_dataset)} tweets")

#################################################################################################################

    else:

        data_collator = DataCollatorMultiTask(tokenizer)

        # Add sentiment and sexism labels
        # Load and process data
        if args.curriculum_learning == True:
            print("Sorting texts based on level of difficulty...")
            # Load and process data
            train_dataset, cl_train_df, train_df, train_max_len = process_dataset(args.train_dataset_path, args.evaluation_type, args.label_column, args.text_column, args.sentiment_column, args.language, tokenizer,  baseline=False, sentiment=True, curriculum_learning=True)
            remove_train_columns = ["id", "tweet", "difficulty", "lang"]
            
        else:   
            # Load and process data
            train_dataset, train_df, train_max_len = process_dataset(args.train_dataset_path, args.evaluation_type, args.label_column, args.text_column, args.sentiment_column, args.language, tokenizer, baseline=False, sentiment=True, curriculum_learning=False)
            remove_train_columns = ["id", "tweet", "lang"]

        test_dataset, test_df, test_max_len = process_dataset(args.test_dataset_path, args.evaluation_type, None, args.text_column, args.sentiment_column, args.language, tokenizer, baseline=False, sentiment=True, curriculum_learning=False)  
        remove_columns = ["id", "tweet", "lang"]
  
        train_class_weights = compute_positive_weights_lang(train_df, LABEL_LIST, label_column="labels_sexist")

        # Tokenize datasets
        encoded_train_dataset = train_dataset.map(
            lambda x: tokenize_sentiment_sexism(x, args.text_column, tokenizer, train_max_len, args),
            batched=True,
            remove_columns=remove_train_columns,
            desc="Tokenizing training dataset..."
        )

        encoded_test_dataset = test_dataset.map(
            lambda x: tokenize_sentiment_sexism(x, args.text_column, tokenizer, test_max_len, args),
            batched=True,
            remove_columns=remove_columns,
            desc="Tokenizing training dataset..."
        )

        columns = ["input_ids", "attention_mask", "language", "labels_sexist", "labels_sentiment"]
        encoded_train_dataset.set_format(type="torch", columns=columns)
        encoded_test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "language", "labels_sentiment"])
        print(f"Train dataset: {len(encoded_train_dataset)} tweets")
        print(f"Test dataset: {len(encoded_test_dataset)} tweets")        

    # Load Model
    if args.model_architecture == "multi-task":

        config = AutoConfig.from_pretrained(
        args.pretrained_model,
        problem_type=args.problem_type,
        id2label=id2label,
        label2id=label2id,
        output_hidden_states=False, 
        output_attentions=False,
        attn_implementation="eager",
        classifier_dropout=args.classifier_dropout if hasattr(args, "classifier_dropout") else 0.1,
        )

        if args.pretrained_model in ["xlm-roberta-base", "xlm-roberta-large", "cardiffnlp/twitter-xlm-roberta-large-2022"]:
            print("\nInitializing XLM-RoBERTa Multihead Multilabel model architecture...")
            print(f'Selected Pooling strategy: {args.pooling_method}')
            model = MultiTask_MultiHead_XLMRoberta.from_pretrained(args.pretrained_model,
                                                                    config = config,
                                                                    pooling_strategy = args.pooling_method,
                                                                    num_sexist_labels = len(LABEL_LIST),
                                                                    num_sentiment_labels = len(SENTIMENT_LABELS))
        else:
            raise ValueError("Please select a valid model architecture: 'xlm-roberta-base', 'xlm-roberta-large' or 'cardiffnlp/twitter-xlm-roberta-large-2022'")
        
        
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {num_trainable}")
        
#################################################################################################################
             
    elif args.model_architecture == "baseline":
        print("Initializing AutoModelForSequenceClassification model architecture for baseline experiments...")
        model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model, 
                                                                num_labels = len(LABEL_LIST), 
                                                                id2label = id2label,
                                                                label2id = label2id,
                                                                problem_type = args.problem_type
                                                                )
    else:
        raise ValueError("Please select a valid model architecture: 'multi-task' or 'baseline'")

    model = model.to(device)

    if args.add_special_tokens == True:
        print('>>>Adding special tokens in tokenizer...')
        additional_special_tokens = ["<user>", "<url>", "<email>", "<date>", "<number>", "<phone>"]
        tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
        model.resize_token_embeddings(len(tokenizer))

#################################################################################################################

    output_dir = args.output_dir + "_" + args.model_architecture + "_" + args.pooling_method + "_" + args.loss_function + "_" + args.language + "_" + args.evaluation_type

    # Define train arguments
    arguments = TrainingArguments(
        output_dir = output_dir,
        logging_dir = f"{output_dir}/logs",
        eval_strategy = args.evaluation_strategy,
        save_strategy = args.save_strategy,
        eval_steps = args.eval_steps,
        save_total_limit = args.save_total_limit,
        learning_rate = args.learning_rate,
        num_train_epochs = args.num_train_epochs,
        per_device_train_batch_size = args.train_batch_size,
        overwrite_output_dir = args.overwrite_output_dir,
        fp16 = args.fp16,
        bf16 = args.bf16,
        fp16_full_eval= args.fp16_full_eval, 
        seed = args.seed,
        # warmup_ratio = args.warmup_ratio, ##
        # weight_decay = args.weight_decay, #
        # adam_epsilon = args.adam_epsilon,
        # max_grad_norm = args.max_grad_norm, #
        max_grad_norm = 1.0,
        optim = args.optimizer, 
        save_steps = args.save_steps,
        logging_strategy = args.logging_strategy,
        logging_steps = args.logging_steps,
        group_by_length = args.group_by_length,
        lr_scheduler_type = args.lr_scheduler_type,
        report_to = args.report_to,
        push_to_hub = args.push_to_hub,
        hub_strategy = args.hub_strategy)

    print(f'Selected Loss function: {args.loss_function}')

    # Create directories to store confusion matrices & results
    check_create_path(f"{output_dir}/conf_matrices/")
    check_create_path(f"{output_dir}/conf_matrices/best_threshs/")
    check_create_path(f"{output_dir}/best_threshs/")
    check_create_path(f"{output_dir}/results/")
    check_create_path(f"{output_dir}/predictions/")
    check_create_path(f"{output_dir}/error_analysis/")

#################################################################################################################
   
    if args.model_architecture == "baseline":
        print("\nSTARTING TRAINING FOR BASELINES...")
        trainer = MultiLabelTrainer(  
                        model = model,
                        data_collator = data_collator,
                        tokenizer = tokenizer,
                        args = arguments,
                        class_weights= train_class_weights, 
                        loss_func_name = args.loss_function,
                        class_frequencies = class_freq, 
                        train_number= train_num,  
                        train_dataset = encoded_train_dataset,        
                        eval_dataset = None,
                        evaluation_type = args.evaluation_type,  
                        compute_metrics = None)

        print("TRAINING MODEL...")
        train_result = trainer.train(resume_from_checkpoint=False)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        metrics = train_result.metrics
        metrics["train_samples"] = len(encoded_train_dataset)
        trainer.log_metrics("train", flatten_dict(metrics))
        trainer.save_metrics("train", flatten_dict(metrics))
        trainer.save_state()

        print("PREDICTING UNLABELLED TEST DATA...")
        test_outputs = trainer.predict(encoded_test_dataset)
        test_logits_sexist = test_outputs.predictions

        with open(args.thresholds_file_path, "r") as f:
            metrics1 = json.load(f)

        # ======Test set======
        test_predictions_json_path = f"{output_dir}/predictions/test_predictions.json"
        test_predictions_json = generate_predictions_file_sexism(
                            logits_sexist = test_logits_sexist,
                            dataframe = test_df,
                            output_json = test_predictions_json_path,
                            categories = LABEL_LIST, 
                            metrics = metrics1,
                            evaluation_type = args.evaluation_type,
                            id_column = "id")

#################################################################################################################
    
    # Use Multi-task Learning
    elif args.model_architecture == "multi-task":
        print("\nSTARTING TRAINING FOR MULTI-TASK LEARNING...")
        trainer = MultiTaskTrainer(
                    model = model,
                    data_collator = data_collator,
                    args = arguments,
                    class_weights = train_class_weights, 
                    train_dataset = encoded_train_dataset,
                    eval_dataset = None,  
                    tokenizer = tokenizer,
                    compute_metrics = None)

        print("TRAINING MODEL...")
        train_result = trainer.train(resume_from_checkpoint=False)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        metrics = train_result.metrics
        metrics["train_samples"] = len(encoded_train_dataset)
        trainer.log_metrics("train", flatten_dict(metrics))
        trainer.save_metrics("train", flatten_dict(metrics))
        trainer.save_state()

        print("PREDICTING UNLABELLED TEST DATA...")
        test_outputs = trainer.predict(encoded_test_dataset)
        test_logits_sexist, test_logits_sentiment, test_language_ids = test_outputs.predictions

        with open(args.thresholds_file_path, "r") as f:
            metrics1 = json.load(f)

        # ======Test set======
        test_predictions_json_path = f"{output_dir}/predictions/test_predictions.json"
        test_predictions_json = generate_predictions_file_sexism_multitask(
                            logits_sexist = test_logits_sexist,
                            dataframe = test_df,
                            output_json = test_predictions_json_path,
                            categories = LABEL_LIST, 
                            metrics = metrics1,
                            evaluation_type = args.evaluation_type,
                            id_column = "id")

        test_multitask_predictions_json_path = f"{output_dir}/predictions/test_multitask_predictions.json"
        test_sexist_sentiment_predictions_json = generate_multitask_predictions_file(
            logits_sexist = test_logits_sexist,
            logits_sentiment = test_logits_sentiment,
            dataframe = test_df,  
            output_json = test_multitask_predictions_json_path,
            sexism_labels = LABEL_LIST,  
            metrics = metrics1,
            evaluation_type = args.evaluation_type,
            id_column = "id")

#################################################################################################################
    else:
        raise ValueError("Please select a valid model architecture: 'multi-task' or 'baseline'")

#################################################################################################################

if __name__ == "__main__":
    args = parse_args()
    main(args)
