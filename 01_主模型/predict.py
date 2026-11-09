#!/usr/bin/env python3
"""Predict syndrome-element sets from a standalone GRU checkpoint."""

import argparse
import csv
import json
from pathlib import Path

import torch

from gru_sequence import SymptomGRU


def batch_tokens(rows, indices):
    selected = [rows[int(index)] for index in indices]
    result = torch.zeros((len(selected), max(map(len, selected))), dtype=torch.long)
    for i, row in enumerate(selected):
        result[i, :len(row)] = torch.tensor(row, dtype=torch.long)
    return result


def read_input(path, schema):
    symptom_index = {name: index + 2
                     for index, name in enumerate(schema["symptoms"])}
    records = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=" ")
        if reader.fieldnames not in (["id", "symptom"], ["id", "symptom", "Se"]):
            raise ValueError("Input must have columns: id symptom [Se]")
        for row in reader:
            symptom_names = [name for name in row["symptom"].split(",") if name]
            unknown = sorted(set(symptom_names) - set(symptom_index))
            if unknown:
                raise ValueError("Unrecognized symptoms in record {}: {}".format(
                    row["id"], ", ".join(unknown)))
            tokens = [symptom_index[name] for name in symptom_names]
            records.append((row["id"], symptom_names, tokens))
    if not records:
        raise ValueError("Input contains no records")
    return records


@torch.no_grad()
def run_prediction(model, tokens):
    model.eval()
    predictions = []
    traces = []
    for start in range(0, len(tokens), 64):
        batch = batch_tokens(tokens, range(start, min(start + 64, len(tokens))))
        batch = batch.to(next(model.parameters()).device)
        predicted, batch_traces = model.generate(batch)
        predictions.extend(predicted.cpu().to(torch.int64).tolist())
        traces.extend(batch_traces)
    return predictions, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", type=Path,
                        default=Path(__file__).resolve().parent / "广安门心绞痛4200例.txt")
    parser.add_argument("--output", type=Path, default=Path("predictions.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    model = SymptomGRU(config["vocab_size"], config["label_count"],
                       embedding=config["embedding"], hidden=config["hidden"],
                       dropout=config["dropout"])
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model.to(device)

    records = read_input(args.input, checkpoint["schema"])
    predicted, traces = run_prediction(model, [row[2] for row in records])
    labels = checkpoint["schema"]["labels"]
    results = []
    for (record_id, symptoms, _), mask, trace in zip(records, predicted, traces):
        results.append({
            "id": record_id,
            "symptom": symptoms,
            "predicted_set": [labels[i] for i, value in enumerate(mask) if value],
            "mask": mask,
            "action_trace": ["STOP" if action == len(labels) else labels[action]
                             for action in trace],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "method": "gru_stop", "fold": checkpoint.get("fold"),
        "seed": checkpoint.get("seed"), "predictions": results,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote {} predictions to {}".format(len(results), args.output.resolve()))


if __name__ == "__main__":
    main()
