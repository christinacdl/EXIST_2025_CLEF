import json
import argparse
from collections import Counter, defaultdict
from utils import evaluate_model_pyeval

SEXISM_HIERARCHY = {
    "YES": [
        "IDEOLOGICAL-INEQUALITY",
        "STEREOTYPING-DOMINANCE",
        "OBJECTIFICATION",
        "SEXUAL-VIOLENCE",
        "MISOGYNY-NON-SEXUAL-VIOLENCE"
    ],
    "NO": []
}

SEXISM_LABELS = SEXISM_HIERARCHY["YES"] + ["NO"]


def load_predictions(filepath):
    with open(filepath, 'r') as f:
        return {entry["id"]: entry["value"] for entry in json.load(f)}


def vote_by_threshold(label_lists, threshold):
    flat_labels = [label for labels in label_lists for label in labels if label in SEXISM_LABELS and label != "NO"]
    if not flat_labels:
        return ["NO"]
    counts = Counter(flat_labels)
    majority_labels = [label for label, count in counts.items() if count >= threshold]
    return majority_labels if majority_labels else ["NO"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="majority_vote_predictions.json")
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--vote_threshold", type=int, default=2)
    parser.add_argument("--input_jsons", nargs='+', required=True, help="List of prediction JSON files to ensemble")
    parser.add_argument("--evaluation_type", type=str, default="hard")
    args = parser.parse_args()

    # Load all predictions
    predictions_by_model = [load_predictions(path) for path in args.input_jsons]

    # Assume all files have same set of IDs in same order
    tweet_ids = list(predictions_by_model[0].keys())

    ensemble_output = []
    for tweet_id in tweet_ids:
        all_labels = [model_preds[tweet_id] for model_preds in predictions_by_model]
        voted_labels = vote_by_threshold(all_labels, args.vote_threshold)
        ensemble_output.append({"test_case": "EXIST2025", "id": tweet_id, "value": voted_labels})

    # Save the final output
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(ensemble_output, f, indent=2, ensure_ascii=False)

    # Evaluate
    results = evaluate_model_pyeval(predictions_json=args.output_file, gold_json=args.gold_file, mode=args.evaluation_type)
    print("\nPyEvALL Evaluation Results:")
    print(results)

if __name__ == "__main__":
    main()
