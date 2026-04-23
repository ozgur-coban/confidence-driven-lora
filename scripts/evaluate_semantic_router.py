"""
Copyright (C) 2026 Özgür ÇOBAN

This file is part of the confidence-driven-lora.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import os
import time
import argparse
from sklearn.metrics import confusion_matrix, classification_report
from src.router.semantic_router import SemanticRouter
from src.data_ops.routing_dataset_manager import DatasetManager
from config.router_dataset_configs import ROUTER_DATASET_CONFIGS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Semantic Router with F1 Scores."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.25, help="Score threshold for routing."
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=1400,
        help="Samples per domain to match Hybrid eval.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("1. Loading Pre-trained Semantic Router...")
    router = SemanticRouter()
    router.load("./models/semantic_centroids.npy")

    print("\n2. Fetching Test Data from Data Manager...")
    general_datasets = ROUTER_DATASET_CONFIGS.get("local_paths")
    domain_rules = ROUTER_DATASET_CONFIGS.get("domain_rules")
    data_manager = DatasetManager(general_datasets, domain_rules)
    test_data = data_manager.get_testing_data(test_samples=args.test_samples)

    print("\n" + "=" * 70)
    print("SEMANTIC ROUTER EVALUATION REPORT")
    print("=" * 70)

    all_y_true = []
    all_y_pred = []

    total_samples = 0
    start_time = time.time()

    for true_domain, prompts in test_data.items():
        for prompt in prompts:
            # Semantic predict returns (final_decision, best_guess, score)
            final_decision, _, _ = router.predict(prompt, threshold=args.threshold)

            all_y_true.append(true_domain)
            all_y_pred.append(final_decision)

        total_samples += len(prompts)

    elapsed = time.time() - start_time
    domain_labels = ["math", "code", "medical", "general"]

    print(
        f"\nEvaluation completed in {elapsed:.2f}s ({total_samples / elapsed:.1f} prompts/sec)"
    )

    print("\n" + "=" * 70)
    print("ROUTER CONFUSION MATRIX")
    print("=" * 70)
    cm = confusion_matrix(all_y_true, all_y_pred, labels=domain_labels)
    header = f"{'True \ Pred':>15} |" + "".join(
        [f"{label:>10}" for label in domain_labels]
    )
    print(header)
    print("-" * len(header))
    for i, true_label in enumerate(domain_labels):
        row_str = f"{true_label:>15} |" + "".join([f"{val:>10}" for val in cm[i]])
        print(row_str)

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT (Precision, Recall, F1)")
    print("=" * 70)
    report = classification_report(
        all_y_true, all_y_pred, labels=domain_labels, target_names=domain_labels
    )
    print(report)


if __name__ == "__main__":
    main()
