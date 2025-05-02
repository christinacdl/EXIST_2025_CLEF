#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,0
CUDA_LAUNCH_BLOCKING=1,0

evaluation_type='hard'
llm_predictions='llm_TEST_predictions_llama3_2_definitions_3b.json'
input_texts_file='EXIST_2025_Tweets_Dataset/mixed/test_no_labels_preprocessed.csv'

python LLM_prompting.py \
    --evaluation_type ${evaluation_type} \
    --llm_predictions ${llm_predictions} \
    --input_texts_file ${input_texts_file}
