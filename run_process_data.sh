#!/bin/bash

train_dir='EXIST_2025_Tweets_Dataset/training'
dev_dir='EXIST_2025_Tweets_Dataset/dev'
test_dir='EXIST_2025_Tweets_Dataset/test'
output_dir='EXIST_2025_Tweets_Dataset/mixed'
language='all'
text_column='tweet'
label_column='hard_label' 
evaluation_type='hard'

python process_data.py \
    --train_dir ${train_dir} \
    --dev_dir ${dev_dir} \
    --test_dir ${test_dir} \
    --output_dir ${output_dir} \
    --language ${language} \
    --text_column ${text_column} \
    --label_column ${label_column} \
    --evaluation_type ${evaluation_type}
