#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
CUDA_LAUNCH_BLOCKING=1

wandb_project='EXIST_CLEF2025'
wandb_name='EXIST_experiments'
pretrained_model='cardiffnlp/twitter-xlm-roberta-large-2022'
train_dataset_path='EXIST_2025_Tweets_Dataset/processed/training_hard_labels.csv'
dev_dataset_path='EXIST_2025_Tweets_Dataset/processed/dev_hard_labels.csv'
test_dataset_path='EXIST_2025_Tweets_Dataset/processed/test_preprocessed.csv'
output_dir='final_experiments/twitter-xlm-roberta-large_multitask_MaxPooling_processed_DBloss_Weights_30_2LAYERS_1e-5' 
val_predictions_file_name='val_preds_05.tsv'
dev_predictions_file_name='test_special_tokens_best_thresholds_predictions.tsv'
test_predictions_file_name='test_preds_05.tsv'
problem_type='multi_label_classification'
evaluation_type='hard'
language='all'
pooling_method='max'
text_column='tweet'
label_column='hard_labels'
sentiment_column='sentiment'
loss_function='DBloss'
report_to='wandb'
evaluation_strategy='epoch'
logging_strategy='epoch'
save_strategy='epoch'
lr_scheduler_type='linear'
optimizer='adamw_torch_fused'
hub_strategy='every_save'
metric_for_best_model='ICM'
padding='max_length'
padding_side='left'
truncation='longest_first'
model_architecture='multi-task'
peft_type='LoRA'
task_type='SEQ_CLS'

python train_model.py \
    --wandb_project ${wandb_project} \
    --wandb_name ${wandb_name} \
    --pretrained_model ${pretrained_model} \
    --train_dataset_path ${train_dataset_path} \
    --dev_dataset_path ${dev_dataset_path} \
    --test_dataset_path ${test_dataset_path} \
    --output_dir ${output_dir} \
    --text_column ${text_column} \
    --label_column ${label_column} \
    --sentiment_column ${sentiment_column} \
    --evaluation_type ${evaluation_type} \
    --pooling_method ${pooling_method} \
    --val_predictions_file_name ${val_predictions_file_name} \
    --dev_predictions_file_name ${dev_predictions_file_name} \
    --test_predictions_file_name ${test_predictions_file_name} \
    --problem_type ${problem_type} \
    --language ${language} \
    --loss_function ${loss_function} \
    --report_to ${report_to} \
    --evaluation_strategy ${evaluation_strategy} \
    --logging_strategy ${logging_strategy} \
    --save_strategy ${save_strategy} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --optimizer ${optimizer} \
    --hub_strategy ${hub_strategy} \
    --metric_for_best_model ${metric_for_best_model} \
    --padding ${padding} \
    --padding_side ${padding_side} \
    --truncation ${truncation} \
    --model_architecture ${model_architecture} \
    --num_labels=6 \
    --seed=2025 \
    --max_seq_length=165 \
    --train_batch_size=32 \
    --val_batch_size=32 \
    --num_train_epochs=30  \
    --warmup_steps=100  \
    --early_stopping_patience=10  \
    --save_steps=500  \
    --logging_steps=100  \
    --eval_steps=500  \
    --save_total_limit=2  \
    --learning_rate=1e-5  \
    --weight_decay=0.01  \
    --warmup_ratio=0.2  \
    --adam_epsilon=1e-8  \
    --data_split_ratio=0.2 \
    --peft_type ${peft_type} \
    --task_type ${task_type} \
    --num_virtual_tokens=241
