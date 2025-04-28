import os
import numpy as np
import pandas as pd
import torch
from transformers import  BitsAndBytesConfig, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, DataCollatorWithPadding
from scipy.special import expit
from datasets import Dataset
from train_arguments import parse_args
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
    set_seed, check_create_path, process_dataset, tokenize, tokenize_sentiment_sexism, compute_positive_weights, compute_train_number, compute_class_frequencies,
    compute_metrics, DataCollatorMultiTask, compute_positive_weights_lang, generate_predictions_file_sexism_multitask,
    MultiTaskTrainer, MultiLabelTrainer, find_best_thresholds, find_best_thresholds_multitask, calculate_metrics, calculate_metrics_perclass_threshs, calculate_metrics_multitask, calculate_metrics_perclass_threshs_multitask,
    perform_error_analysis, perform_error_analysis_multitask, generate_predictions_file_sexism,  load_best_thresholds_from_checkpoint, generate_multitask_predictions_file, compute_metrics_multitask, write_tsv_dataframe, train_evaluate_predict, generate_predictions_file, evaluate_model_pyeval
)
# watch -n 60 nvidia-smi
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
                                            #   padding_side = "right",
                                            #   trust_remote_code=True,
                                            #   add_eos_token=True
                                              )  
    ## PEFT LoRA
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.pad_token = tokenizer.eos_token

    # prompt_tuning_init_text="""[INST]"""

    # tokens = tokenizer.tokenize(prompt_tuning_init_text)  # For PEFT only
    # num_tokens = len(tokens)
    # print(f"The prompt contains {num_tokens} tokens.")
    
    # compute_dtype = getattr(torch, "float16")
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=compute_dtype)

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

        dev_dataset, dev_df, dev_max_len = process_dataset(args.dev_dataset_path, args.evaluation_type, args.label_column, args.text_column, args.sentiment_column, args.language, tokenizer,  baseline=True, sentiment=False, curriculum_learning=False)
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

        encoded_dev_dataset = dev_dataset.map(
            lambda x: tokenize(x, args.text_column, tokenizer, dev_max_len, args),
            batched=True,
            remove_columns=remove_columns,
            load_from_cache_file=False,
            desc="Running tokenizer on development dataset..."
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
        encoded_dev_dataset.set_format(type="torch", columns=columns)
        encoded_test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        print(f"Train dataset: {len(encoded_train_dataset)} tweets")
        print(f"Development dataset: {len(encoded_dev_dataset)} tweets")
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

        dev_dataset, dev_df, dev_max_len = process_dataset(args.dev_dataset_path, args.evaluation_type, args.label_column, args.text_column, args.sentiment_column, args.language, tokenizer, baseline=False, sentiment=True, curriculum_learning=False)
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

        encoded_dev_dataset = dev_dataset.map(
            lambda x: tokenize_sentiment_sexism(x, args.text_column, tokenizer, dev_max_len, args),
            batched=True,
            remove_columns=remove_columns,
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
        encoded_dev_dataset.set_format(type="torch", columns=columns)
        encoded_test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "language", "labels_sentiment"])

        print(f"Train dataset: {len(encoded_train_dataset)} tweets")
        print(f"Development dataset: {len(encoded_dev_dataset)} tweets")
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
                                                                problem_type = args.problem_type,
                                                                # quantization_config = bnb_config,
                                                                # device_map="auto",
                                                                # offload_folder = 'offload',
                                                                # trust_remote_code = True, 
                                                                # torch_dtype = torch.float16
                                                                )
    else:
        raise ValueError("Please select a valid model architecture: 'multi-task' or 'baseline'")


    model = model.to(device)
    
    # model.config.use_cache = False
    # model.config.pretraining_tp = 1
    # model.config.pad_token_id = model.config.eos_token_id
    # model.config.pad_token_id = tokenizer.pad_token_id

    if args.add_special_tokens_context == True:
        print('>>>Adding special tokens in tokenizer...')
        additional_special_tokens = ["<user>", "<url>", "<email>", "<date>", "<number>", "<phone>"]
        tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
        model.resize_token_embeddings(len(tokenizer))

    # peft_config = LoraConfig(lora_alpha = 16, 
    #                        lora_dropout = 0.1, 
    #                        r = 8, 
    #                        bias = 'none',
    #                        target_modules = ['q_proj','v_proj'],
    #                        task_type = args.task_type,
    #                        inference_mode=False)

#################################################################################################################
                             
    output_dir = args.output_dir + "_" + args.language + "_" + args.evaluation_type
    #+ "_" + args.model_architectur+ "_" + args.pooling_method + "_" + args.loss_function 
    

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
        metric_for_best_model = args.metric_for_best_model,
        greater_is_better = args.greater_is_better,
        load_best_model_at_end = args.load_best_model_at_end,
        per_device_train_batch_size = args.train_batch_size,
        per_device_eval_batch_size = args.val_batch_size,
        overwrite_output_dir = args.overwrite_output_dir,
        fp16 = args.fp16,
        bf16 = args.bf16,
        fp16_full_eval= args.fp16_full_eval, 
        seed = args.seed,
        # warmup_ratio = args.warmup_ratio, ##
        # weight_decay = args.weight_decay, #
        # adam_epsilon = args.adam_epsilon,
        max_grad_norm = 1.0,
        # gradient_checkpointing=True,
        # max_grad_norm = args.max_grad_norm, #
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
                        class_weights= None, #train_class_weights, 
                        loss_func_name = args.loss_function,
                        class_frequencies = class_freq, 
                        train_number= train_num,  
                        train_dataset = encoded_train_dataset,        
                        eval_dataset = encoded_dev_dataset,
                        evaluation_type = args.evaluation_type,  
                        compute_metrics = lambda e: compute_metrics(eval_pred=e, NUM_LABELS=len(LABEL_LIST), label_list=LABEL_LIST),
                        callbacks = [EarlyStoppingCallback(early_stopping_patience = args.early_stopping_patience)])

        logits_sexist, labels_sexist, test_logits_sexist = train_evaluate_predict(trainer, model, tokenizer, output_dir, 
                                                             encoded_train_dataset, encoded_dev_dataset, 
                                                             encoded_test_dataset = encoded_test_dataset, multitask=False)

        best_checkpoint = trainer.state.best_model_checkpoint
        if best_checkpoint is not None:
            print(f"[INFO] Using best model checkpoint: {best_checkpoint}")
            saved_thresholds_path = os.path.join(best_checkpoint, "best_thresholds.json")
            saved_thresholds_path = os.path.join(output_dir, "best_threshs/thresholds.json")
        else:
            print("[WARNING] No best checkpoint found. Using current output_dir.")
            saved_thresholds_path = os.path.join(output_dir, "best_threshs/thresholds.json")

        metrics1, best_thresh = find_best_thresholds(
            predictions = logits_sexist,
            labels = labels_sexist,
            dataframe = dev_df,
            label_list = LABEL_LIST,
            output_json = saved_thresholds_path
        )

        # === Calculate performance metrics on labelled validation/development set === #
        # Threshold 1: Fixed threshold (0.5)
        print("\nEvaluating with threshold = 0.5")
        val_scores1 = calculate_metrics(
            labels_sexist, logits_sexist, LABEL_LIST,
            save_directory_name1 = f"{output_dir}/results/report_0.5.txt",
            save_directory_name2 = f"{output_dir}/conf_matrices/val_dataset_0.5",
            save_directory_name3 = f"{output_dir}/val_dataset_0.5_matrix.png",
            thresh = 0.5, sigmoid = True)
        write_tsv_dataframe(os.path.join(f"{output_dir}/results/", "val_scores_0.5.tsv"), val_scores1)

        # Threshold 2: Best threshold for all labels
        print("\nEvaluating with best threshold for all labels:", best_thresh)
        val_scores2 = calculate_metrics(
            labels_sexist, logits_sexist, LABEL_LIST,
            save_directory_name1 = f"{output_dir}/results/report_{best_thresh}.txt",
            save_directory_name2 = f"{output_dir}/conf_matrices/val_dataset_{best_thresh}",
            save_directory_name3 = f"{output_dir}/val_dataset_{best_thresh}_matrix.png",
            thresh = best_thresh, sigmoid = True)
        write_tsv_dataframe(os.path.join(f"{output_dir}/results/", f"val_scores_{best_thresh}.tsv"), val_scores2)

        # Threshold 3: Best threshold for each label
        print("\nEvaluating with best threshold per label")
        val_scores3 = calculate_metrics_perclass_threshs(
            labels_sexist, logits_sexist, LABEL_LIST,
            save_directory_name1 = f"{output_dir}/results/report_best_threshs.txt",
            save_directory_name2 = f"{output_dir}/conf_matrices/best_threshs/val_dataset_best_threshs",
            save_directory_name3 = f"{output_dir}/val_dataset_best_threshs_matrix.png",
            best_thresholds_f1_scores = metrics1, sigmoid = True)
        write_tsv_dataframe(os.path.join(f"{output_dir}/results/", "val_scores_best_threshs.tsv"), val_scores3)

        # ======Development set======        
        predictions_json_path = f"{output_dir}/predictions/val_predictions.json"
        predictions_json = generate_predictions_file_sexism(
                            logits_sexist = logits_sexist,
                            dataframe = dev_df,
                            output_json = predictions_json_path,
                            categories = LABEL_LIST, 
                            metrics = metrics1,
                            evaluation_type = args.evaluation_type,
                            id_column = "id")
      
        if args.evaluation_type == "hard":
            gold_json = "EXIST_2025_Tweets_Dataset/dev/EXIST2025_dev_task1_3_gold_hard.json"
        elif args.evaluation_type == "soft":
            gold_json = "EXIST_2025_Tweets_Dataset/dev/EXIST2025_dev_task1_3_gold_soft.json"
        else:
            raise ValueError(f"Please provide the true labels json file based on evaluation type: {args.evaluation_type}.")
        
        # PyEval Evaluation on development set
        pyeval_results = evaluate_model_pyeval(predictions_json, gold_json, mode = args.evaluation_type)
        print("\nPyEval Evaluation Results:")
        print(pyeval_results)

        with open(f"{output_dir}/results/PyEval_report_dev.json", "w") as f:
            json.dump(pyeval_results, f, indent=4)

        # Perform Error Analysis on development set
        error_results, no_pred_true, no_true_pred = perform_error_analysis(
            val_labels = labels_sexist,
            val_dataframe = dev_df,
            prediction_json_path = predictions_json_path,
            label_list = LABEL_LIST)

        with open(f"{output_dir}/error_analysis/error_analysis.json", "w") as f:
            json.dump(error_results, f, indent=4)

        with open(f"{output_dir}/error_analysis/no_pred_true.json", "w") as f:
            json.dump(no_pred_true, f, indent=4)

        with open(f"{output_dir}/error_analysis/no_true_pred.json", "w") as f:
            json.dump(no_true_pred, f, indent=4)
        print("\nError analysis results saved!")


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
                    eval_dataset = encoded_dev_dataset,  
                    tokenizer = tokenizer,
                    compute_metrics = lambda e: compute_metrics_multitask(eval_pred=e, label_list=LABEL_LIST),
                    callbacks = [EarlyStoppingCallback(early_stopping_patience = args.early_stopping_patience)])

        logits_sexist, logits_sentiment, labels_sexist, labels_sentiment, language_ids, test_logits_sexist, test_logits_sentiment, test_language_ids = train_evaluate_predict(trainer, model, tokenizer, output_dir, 
                                                                                                                                                                            encoded_train_dataset, encoded_dev_dataset, 
                                                                                                                                                                            encoded_test_dataset=encoded_test_dataset, multitask=True)

        best_checkpoint = trainer.state.best_model_checkpoint
        if best_checkpoint is not None:
            print(f"[INFO] Using best model checkpoint: {best_checkpoint}")
            saved_thresholds_path = os.path.join(best_checkpoint, "best_thresholds.json")
            saved_thresholds_path = os.path.join(output_dir, "best_threshs/thresholds.json")
        else:
            print("[WARNING] No best checkpoint found. Using current output_dir.")
            saved_thresholds_path = os.path.join(output_dir, "best_threshs/thresholds.json")

        metrics1, best_thresh = find_best_thresholds_multitask(
            sexist_logits=logits_sexist,
            sexist_labels=labels_sexist,
            sentiment_logits = logits_sentiment,
            dataframe=dev_df, 
            label_list=LABEL_LIST,
            output_json=saved_thresholds_path
        )

        # === Calculate performance metrics on labelled validation/development set === #
        # Threshold 1: Fixed threshold (0.5)
        print("\nEvaluating with threshold = 0.5")
        val_scores1 = calculate_metrics_multitask(
            y_true_sexist = labels_sexist,
            y_pred_sexist = logits_sexist,
            y_true_sentiment = labels_sentiment,
            y_pred_sentiment = logits_sentiment,
            class_names = LABEL_LIST,
            sentiment_labels = SENTIMENT_LABELS,
            save_directory_name1 = f"{output_dir}/results/report_0.5.txt",
            save_directory_name2 = f"{output_dir}/conf_matrices/val_dataset_0.5",
            save_directory_name3 = f"{output_dir}/val_dataset_0.5_matrix.png",
            save_directory_name4 = f"{output_dir}/results/sentiment_report_0.5.txt",
            thresh = 0.5, sigmoid = True)
        write_tsv_dataframe(os.path.join(f"{output_dir}/results/", "val_scores_0.5.tsv"), val_scores1)

        # Threshold 2: Best threshold for all labels
        print("\nEvaluating with best threshold for all labels:", best_thresh)
        val_scores2 = calculate_metrics_multitask(
            y_true_sexist = labels_sexist,
            y_pred_sexist = logits_sexist,
            y_true_sentiment = labels_sentiment,
            y_pred_sentiment = logits_sentiment,
            class_names = LABEL_LIST,
            sentiment_labels = SENTIMENT_LABELS,
            save_directory_name1 = f"{output_dir}/results/report_{best_thresh}.txt",
            save_directory_name2 = f"{output_dir}/conf_matrices/val_dataset_{best_thresh}",
            save_directory_name3 = f"{output_dir}/val_dataset_{best_thresh}_matrix.png",
            save_directory_name4 = f"{output_dir}/results/sentiment_report_{best_thresh}.txt",
            thresh = best_thresh, sigmoid = True)
        write_tsv_dataframe(os.path.join(f"{output_dir}/results/", f"val_scores_{best_thresh}.tsv"), val_scores2)

        # Threshold 3: Best threshold for each label
        print("\nEvaluating with best threshold per label")
        val_scores3 = calculate_metrics_perclass_threshs_multitask(
            y_true_sexist = labels_sexist,
            y_pred_sexist = logits_sexist,
            y_true_sentiment = labels_sentiment,
            y_pred_sentiment = logits_sentiment,
            class_names = LABEL_LIST,
            sentiment_labels = SENTIMENT_LABELS,
            languages = dev_df["lang"].tolist(),
            save_directory_name1 = f"{output_dir}/results/report_best_threshs.txt",
            save_directory_name2 = f"{output_dir}/conf_matrices/best_threshs/val_dataset_best_threshs",
            save_directory_name3 = f"{output_dir}/val_dataset_best_threshs_matrix.png",
            save_directory_name4 = f"{output_dir}/results/sentiment_report_best_threshs.txt",
            best_thresholds_f1_scores = metrics1, sigmoid = True)
        write_tsv_dataframe(os.path.join(f"{output_dir}/results/", "val_scores_best_threshs.tsv"), val_scores3)

        # ======Development set======
        predictions_json_path = f"{output_dir}/predictions/val_predictions.json"
        predictions_json = generate_predictions_file_sexism_multitask(
                            logits_sexist = logits_sexist,
                            dataframe = dev_df,
                            output_json = predictions_json_path,
                            categories = LABEL_LIST, 
                            metrics = metrics1,
                            evaluation_type = args.evaluation_type,
                            id_column = "id")
        
        if args.evaluation_type == "hard":
            gold_json = "EXIST_2025_Tweets_Dataset/dev/EXIST2025_dev_task1_3_gold_hard.json"
        elif args.evaluation_type == "soft":
            gold_json = "EXIST_2025_Tweets_Dataset/dev/EXIST2025_dev_task1_3_gold_soft.json"
        else:
            raise ValueError(f"Please provide the true labels json file based on evaluation type: {args.evaluation_type}.")
        
        # PyEval Evaluation on development set
        pyeval_results = evaluate_model_pyeval(predictions_json, gold_json, mode = args.evaluation_type)
        print("\nPyEval Evaluation Results:")
        print(pyeval_results)

        with open(f"{output_dir}/results/PyEval_report_dev.json", "w") as f:
            json.dump(pyeval_results, f, indent=4)

        multitask_predictions_json_path = f"{output_dir}/predictions/val_multitask_predictions.json"
        sexist_sentiment_predictions_json = generate_multitask_predictions_file(
            logits_sexist = logits_sexist,
            logits_sentiment= logits_sentiment,
            dataframe = dev_df,  
            output_json = multitask_predictions_json_path,
            sexism_labels = LABEL_LIST,  
            metrics = metrics1,
            evaluation_type = args.evaluation_type,
            id_column = "id")

        # Perform Error Analysis on development set
        error_results, no_pred_true, no_true_pred = perform_error_analysis_multitask(
            val_labels_sexist = labels_sexist, 
            val_labels_sentiment = labels_sentiment, 
            val_dataframe = dev_df, 
            prediction_json_path = multitask_predictions_json_path, 
            label_list = LABEL_LIST, 
            sentiment_labels= SENTIMENT_LABELS)

        with open(f"{output_dir}/error_analysis/error_analysis.json", "w") as f:
            json.dump(error_results, f, indent=4)

        with open(f"{output_dir}/error_analysis/no_pred_true.json", "w") as f:
            json.dump(no_pred_true, f, indent=4)

        with open(f"{output_dir}/error_analysis/no_true_pred.json", "w") as f:
            json.dump(no_true_pred, f, indent=4)
        print("\nError analysis results saved!")


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

        pooling_strategies = ["cls", "mean", "max", "hybrid", "mean_max"]
        print("\n=== Comparing Pooling Strategies ===")

        for strategy in pooling_strategies:
            print(f"\n== Testing pooling strategy: {strategy.upper()} ==")
            
            # Update model heads with new strategy
            for head in model.sexist_heads.values():
                head.pooling_strategy = strategy
            for head in model.sentiment_heads.values():
                head.pooling_strategy = strategy

            # Evaluate with this strategy
            results = trainer.evaluate(metric_key_prefix=strategy)
            print(results)

            # Generate prediction JSON with this strategy
            strategy_json_path = os.path.join(output_dir, f"predictions/val_predictions_{strategy}.json")
            generate_predictions_file_sexism_multitask(
                logits_sexist=logits_sexist,
                dataframe=dev_df,
                output_json=strategy_json_path,
                categories=LABEL_LIST,
                metrics=metrics1,
                evaluation_type=args.evaluation_type,
                id_column="id"
            )

            # Evaluate with PyEvALL
            pyeval_results = evaluate_model_pyeval(strategy_json_path, gold_json, mode=args.evaluation_type)
            print(f"\nPyEvALL Results ({strategy}):")
            print(pyeval_results)

#################################################################################################################
    
    else:
        raise ValueError("Please select a valid model architecture: 'multi-task' or 'baseline'")


if __name__ == "__main__":
    args = parse_args()
    main(args)
