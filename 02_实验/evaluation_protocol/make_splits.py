"""Freeze cardinality-stratified record-level internal CV splits."""

import argparse
import csv
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "广安门心绞痛4200例.txt"
OUTER_SEED = 20260928
INNER_SEED = 20260929
FOLDS = 3
VALIDATION_PERCENT = 15


def read_data(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=" ")
        if reader.fieldnames != ["id", "symptom", "Se"]:
            raise ValueError("Expected columns: id symptom Se")
        rows = list(reader)
    if len(rows) != 4200:
        raise ValueError("Expected 4200 records")

    cardinality = {}
    symptoms = set()
    labels = set()
    for row in rows:
        record_id = int(row["id"])
        symptom_names = row["symptom"].split(",")
        label_names = row["Se"].split(",")
        if not symptom_names or not label_names or "" in symptom_names + label_names:
            raise ValueError(f"Empty token in record {record_id}")
        if len(label_names) != len(set(label_names)):
            raise ValueError(f"Repeated label in record {record_id}")
        cardinality[record_id] = len(label_names)
        symptoms.update(symptom_names)
        labels.update(label_names)
    if sorted(cardinality) != list(range(1, 4201)):
        raise ValueError("Expected unique record ids 1..4200")
    return cardinality, sorted(symptoms), sorted(labels)


def outer_assignment(cardinality):
    by_size = defaultdict(list)
    for record_id, count in cardinality.items():
        by_size[count].append(record_id)
    rng = random.Random(OUTER_SEED)
    assignment = {}
    offset = 0
    for count in sorted(by_size):
        ids = sorted(by_size[count])
        rng.shuffle(ids)
        for position, record_id in enumerate(ids):
            assignment[record_id] = (offset + position) % FOLDS
        offset = (offset + len(ids)) % FOLDS
    return assignment


def validation_ids(cardinality, outer_assignment_by_id, held_out_fold):
    by_size = defaultdict(list)
    for record_id, count in cardinality.items():
        if outer_assignment_by_id[record_id] != held_out_fold:
            by_size[count].append(record_id)

    target = sum(map(len, by_size.values())) * VALIDATION_PERCENT // 100
    quotas = {count: len(ids) * VALIDATION_PERCENT // 100
              for count, ids in by_size.items()}
    extra = target - sum(quotas.values())
    order = sorted(by_size, key=lambda count:
                   (-(len(by_size[count]) * VALIDATION_PERCENT % 100), count))
    for count in order[:extra]:
        quotas[count] += 1

    rng = random.Random(INNER_SEED + held_out_fold)
    chosen = set()
    for count in sorted(by_size):
        ids = sorted(by_size[count])
        rng.shuffle(ids)
        chosen.update(ids[:quotas[count]])
    if len(chosen) != target:
        raise AssertionError("Wrong validation size")
    return chosen


def locked_write(path, content):
    payload = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Locked file differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def build(data_path, output_dir):
    cardinality, symptoms, labels = read_data(data_path)
    outer = outer_assignment(cardinality)
    all_ids = set(cardinality)
    if set(outer) != all_ids:
        raise AssertionError("Outer folds do not cover every record")

    fold_info = []
    split_contents = []
    for fold in range(FOLDS):
        val = validation_ids(cardinality, outer, fold)
        test = {record_id for record_id in all_ids if outer[record_id] == fold}
        train = all_ids - val - test
        if val & test or train & val or train & test or train | val | test != all_ids:
            raise AssertionError("Fold partitions overlap or omit records")
        if (len(train), len(val), len(test)) != (2380, 420, 1400):
            raise AssertionError("Unexpected fold sizes")

        roles = {record_id: ("test" if record_id in test else
                             "val" if record_id in val else "train")
                 for record_id in sorted(all_ids)}
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["id", "split"])
        writer.writerows((record_id, roles[record_id]) for record_id in sorted(all_ids))
        content = buffer.getvalue()
        name = f"fold_{fold + 1}.csv"
        split_contents.append((name, content))
        fold_info.append({
            "fold": fold + 1,
            "file": name,
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "sizes": {role: len(ids) for role, ids in
                      (("train", train), ("val", val), ("test", test))},
            "cardinality": {role: dict(sorted(Counter(cardinality[i] for i in ids).items()))
                            for role, ids in
                            (("train", train), ("val", val), ("test", test))},
        })

    if any(sum(outer[record_id] == fold for fold in range(FOLDS)) != 1
           for record_id in all_ids):
        raise AssertionError("Each record must be tested in exactly one fold")

    protocol = {
        "name": "post-development record-level internal three-fold CV",
        "data_file": data_path.name,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "record_count": len(cardinality),
        "outer_folds": FOLDS,
        "outer_seed": OUTER_SEED,
        "inner_seed": INNER_SEED,
        "inner_validation_percent_of_outer_train": VALIDATION_PERCENT,
        "training_seeds": [9, 17, 29],
        "epochs": 100,
        "batch_size": 64,
        "primary_metric": "exact match",
        "secondary_metric": "sample F1",
        "checkpoint_order": ["validation exact match", "validation sample F1",
                             "earlier epoch"],
        "schema": {"symptoms": symptoms, "labels": labels},
        "folds": fold_info,
    }
    for name, content in split_contents:
        locked_write(output_dir / name, content)
    locked_write(output_dir / "protocol.json",
                 json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    print(f"LOCKED {FOLDS} folds, {len(cardinality)} records, {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "splits")
    args = parser.parse_args()
    build(args.data, args.output_dir)


if __name__ == "__main__":
    main()
