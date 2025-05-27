# EXIST_2025_CLEF

# My participation in Sub-task 1.3 of EXIST (sEXism Identification in Social neTworks)

Step 1: ```bash run_process_data.sh``` => It runs ```process_data.py``` to load and pre-process all the data

Step 2: ```bash run_train_model.sh``` => It runs ```train_model.py``` to fine-tune either baseline models with the standard architecture or the multi-task model architecture and evaluate labelled dev set with metrics 

Step 3: ```bash run_LLM.sh``` => It runs the ```LLM_prompting.py``` to prompt an LLM to annotate with the sexism labels the input data

Step 4: ```bash run_LLM_RAG.sh``` => It runs the ```rag.py``` to prompt an LLM to annotate with the sexism labels the input data using chain-of-thought reasoning and RAG for contextual information

Step 5: ```bash run_majority_vote.sh``` => It runs the ```majority_vote.py``` to create majority predictions based on the models' level of agreement

Step 6: ```bash run_train_model2.sh``` => It runs ```train_model2.py``` to ONLY fine-tune either baseline models with the standard architecture or the multi-task model architecture without any evaluation
