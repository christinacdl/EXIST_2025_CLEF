#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,0
CUDA_LAUNCH_BLOCKING=1,0

evaluation_type='hard'
llm_predictions='llm_predictions_llama32_3B.json'

python LLM_prompting.py \
    --evaluation_type ${evaluation_type} \
    --llm_predictions ${llm_predictions}
