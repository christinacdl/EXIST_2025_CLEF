#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
CUDA_LAUNCH_BLOCKING=1

evaluation_type='hard'
input_jsons='
final_experiments/twitter-xlm-roberta-large_multitask_MaxPooling_processed_BCE_Weights_30_1layer_1e-5_all_hard/predictions/val_predictions.json
final_experiments/twitter-xlm-roberta-large_multitask_MaxPooling_2layers_processed_BCE_Weights_30_1layer_1e-5_all_hard/predictions/val_predictions.json
final_experiments/twitter-xlm-roberta-large_multitask_MaxPooling_3layers_processed_BCE_Weights_30_1layer_1e-5_all_hard/predictions/val_predictions.json
final_experiments/twitter-xlm-roberta-large_multitask_MaxPooling_4layers_processed_BCE_Weights_30_1layer_1e-5_all_hard/predictions/val_predictions.json
final_experiments/twitter-xlm-roberta-large_multitask_CLSPooling_processed_BCE_Weights_30_1layer_1e-5_all_hard/predictions/val_predictions.json
final_experiments/twitter-xlm-roberta-large_multitask_MeanPooling_processed_BCE_Weights_30_1layer_1e-5_all_hard/predictions/val_predictions.json
'
gold_file='EXIST_2025_Tweets_Dataset/dev/EXIST2025_dev_task1_3_gold_hard.json'
output_file='6_majority_vote_max_mean_cls_BCE_DBloss_NEW_predictions.json'


python majority_vote.py \
  --input_jsons ${input_jsons} \
  --vote_threshold=3 \
  --gold_file ${gold_file} \
  --output_file ${output_file}
