import json
import json_repair
import torch
import os
import pandas as pd
from tqdm import tqdm
import argparse
from collections import Counter
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from utils import evaluate_model_pyeval, set_seed
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using GPU is CUDA:{os.environ['CUDA_VISIBLE_DEVICES']}")

os.environ['HF_HOME'] = '/home/ch.christodoulou/classification/CLEF2025_EXIST/hf_cache'

hierarchy = {
    "YES": ["IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE", "OBJECTIFICATION",
            "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE"],
    "NO": []
}

SEXISM_LABELS = hierarchy["YES"] + ["NO"]

class LabelOutput(BaseModel):
    labels: list[str] = Field(description="List of sexism labels")

prompt_template = PromptTemplate(
    input_variables=["tweet", "sentiment_prediction", "retrieved_context", "annotator_profile"],
    template="""
[INST]
You are a sexism detection assistant analyzing tweets in English or Spanish. You have the perspective of the following person:

{annotator_profile}

Each tweet is accompanied by its sentiment.
You are also given examples of previously labeled tweets to help guide your classification.

Before making a decision, think step-by-step:
- Does the tweet express sexist stereotypes, gender roles, objectification, or violence?
- Does it refer to power, bodies, or inequality?
- Consider sentiment and examples carefully.

Tweet:
"{tweet}"

Sentiment: {sentiment_prediction}

Context examples from labeled tweets:
{retrieved_context}

### Sexism Categories (with definitions):
1. **IDEOLOGICAL-INEQUALITY**: The text discredits the feminist movement, rejects inequality between men and women, or presents men as victims of gender-based oppression.
2. **STEREOTYPING-DOMINANCE**: The text expresses false ideas about women that suggest they are more suitable to fulfill certain roles (mother, wife, caregiver, submissive, etc.), or inappropriate for certain tasks (e.g., driving, hard work), or claims that men are superior.
3. **OBJECTIFICATION**: The text presents women as objects, disregarding their dignity and personality, or assumes physical traits women must have to fulfill traditional gender roles (beauty standards, hypersexualization, women’s bodies at men’s disposal, etc.).
4. **SEXUAL-VIOLENCE**: The text includes or describes sexual suggestions, requests for sexual favors, or harassment of a sexual nature, including rape or sexual assault.
5. **MISOGYNY-NON-SEXUAL-VIOLENCE**: The text expresses hatred or non-sexual violence toward women (e.g., insults, aggression, or psychological abuse without sexual undertone).
6. **NO**: Use this only if none of the above categories are present.

### Output Format:
- Return only valid JSON exactly in the format shown below.
- Do not include explanations or extra text.

```json
{{
  "labels": ["<CATEGORY1>", "<CATEGORY2>", ...]
}}

Answer:
[/INST]
"""
)


def summarize_profile(row):
    def most_common(lst):
        return Counter(lst).most_common(1)[0][0] if lst else None

    genders = eval(row['gender_annotators'])
    ages = eval(row['age_annotators'])
    ethnicities = eval(row['ethnicities_annotators'])
    educations = eval(row['study_levels_annotators'])
    countries = eval(row['countries_annotators'])

    gender = most_common(genders)
    age = most_common(ages)
    ethnicity = most_common(ethnicities)
    education = most_common(educations)
    country = most_common(countries)

    gender_text = "female" if gender == "F" else "male"
    profile_str = f"{gender_text}, {age} years old, {ethnicity}, {education}, living in {country}"
    return profile_str

def try_extract_labels(raw_response):
    try:
        json_str = raw_response[raw_response.find("{"): raw_response.rfind("}") + 1]
        print("\n[ORIGINAL LLM OUTPUT]\n" + json_str)
        repaired = json_repair.repair_json(json_str)
        print("\n[REPAIRED JSON]\n" + (repaired if isinstance(repaired, str) else json.dumps(repaired, indent=2)))
        if isinstance(repaired, list):
            response_dict = next((item for item in repaired if isinstance(item, dict) and "labels" in item), {})
        else:
            response_dict = json.loads(repaired)
        labels = response_dict["labels"] if isinstance(response_dict, dict) and "labels" in response_dict else []
        return [label for label in labels if label in SEXISM_LABELS]
    except Exception as e:
        print(f"[!] JSON parsing failed: {e}")
        return []

def compute_prompt_length(input_data, tokenizer):
    formatted_prompt = prompt_template.format(**input_data)
    tokens = tokenizer(formatted_prompt, return_tensors="pt")
    return len(tokens.input_ids[0])

def embed_train_data_for_rag(train_df, embedding_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
    print("[INFO] Embedding training data for ChromaDB...")
    embed_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    documents = [
    Document(
    page_content=f"Tweet: {row['tweet']}\nSentiment: {row['sentiment']}\nLabels: {row['hard_label']}",
    metadata={"id": str(row["id"])}
    ) for _, row in train_df.iterrows()
    ]
    db = Chroma.from_documents(documents, embedding=embed_model, persist_directory="chroma_rag_db")
    return db

def main():
    set_seed(2025)

    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument("--evaluation_type", type=str, default="hard", choices=["hard", "soft"], required=True, help="Evaluation type: hard or soft")
    parser_arg.add_argument("--input_texts_file", type=str, required=True, help="Path to the input texts file")
    parser_arg.add_argument("--train_data", type=str, required=True)
    parser_arg.add_argument("--llm_predictions", type=str, default="llm_predictions.json", required=True, help="Path to save the LLM predictions")
    args = parser_arg.parse_args()

    train_df = pd.read_csv(args.train_data, sep=",", encoding="utf-8")
    db = embed_train_data_for_rag(train_df)
    dev_df = pd.read_csv(args.input_texts_file, sep=",", encoding="utf-8")
    dev_df["annotator_profile"] = dev_df.apply(summarize_profile, axis=1)

    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=False, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", quantization_config=bnb_config) #

    def create_dynamic_pipeline(max_new_tokens):
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens)
        return HuggingFacePipeline(pipeline=pipe)

    llm_only_output = []

    for _, row in tqdm(dev_df.iterrows(), total=len(dev_df), desc="Classifying with LLM"):
        tweet_id = str(row["id"])
        print(f"[DEBUG] Processing tweet ID {tweet_id}")
        tweet = row["tweet"]
        sentiment_prediction = row.get("sentiment")
        annotator_profile = row["annotator_profile"]

        # RAG retrieval
        query = f"{tweet} {sentiment_prediction}"
        retrieved_docs = db.similarity_search_with_score(query, k=50)
        # Filter: must contain at least one valid sexism label
        filtered_docs = [
            doc for doc, score in retrieved_docs
            if any(label in doc.page_content for label in SEXISM_LABELS)]

        # Select top-k filtered examples
        top_k_docs = filtered_docs[:3]
        retrieved_context = "\n\n".join([
            f"Example {i+1}:\nTweet: {doc.page_content.splitlines()[0].replace('Tweet: ', '')}\n"
            f"Sentiment: {doc.page_content.splitlines()[1].replace('Sentiment: ', '')}\n"
            f"Labels: {doc.page_content.splitlines()[2].replace('Labels: ', '')}"
            for i, doc in enumerate(top_k_docs)
        ])

        input_data = {"tweet": tweet, 
        "sentiment_prediction": sentiment_prediction, 
        "retrieved_context": retrieved_context,
        "annotator_profile": annotator_profile}

        prompt_len = compute_prompt_length(input_data, tokenizer)
        max_model_tokens = 4096
        available_tokens = max_model_tokens - prompt_len
        max_new_tokens = min(1024, max(16, available_tokens))

        llm = create_dynamic_pipeline(max_new_tokens)
        chain = prompt_template | llm.bind(skip_prompt=True)

        # print(chain)
        # print(f"[DEBUG] Prompt length: {prompt_len} tokens")
        # print(f"[DEBUG] Max new tokens: {max_new_tokens}")

        try:
            labels = try_extract_labels(chain.invoke(input_data))
            labels = labels if labels else ["NO"]
        except Exception as e:
            print(f"[ERROR] Failed on ID {tweet_id}: {e}")
            labels = ["NO"]

        llm_only_output.append({"test_case": "EXIST2025", "id": tweet_id, "value": labels})

    with open(args.llm_predictions, "w", encoding="utf-8") as f:
        json.dump(llm_only_output, f, indent=2, ensure_ascii=False)

    gold_json = f"EXIST_2025_Tweets_Dataset/dev/EXIST2025_dev_task1_3_gold_{args.evaluation_type}.json"
    results = evaluate_model_pyeval(predictions_json=args.llm_predictions, gold_json=gold_json, mode=args.evaluation_type)

    print("\nPyEvALL Evaluation Results:")
    print(results)

if __name__ == "__main__":
    main()
