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

import json
from collections import defaultdict


def main():
    log_path = "logs/hybrid_router_errors_prioritized_95syntax.json"

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            errors = json.load(f)
    except FileNotFoundError:
        print(
            f"Error: Could not find {log_path}. Ensure you ran evaluate_hybrid.py first."
        )
        return

    print("=" * 90)
    print(f"HYBRID ERROR ANALYSIS ({len(errors)} Total Errors)")
    print("=" * 90)

    # Structure: dict[true_domain][pred_domain][rule] = list of error dicts
    analysis = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for err in errors:
        true_domain = err.get("true_domain", "unknown")
        pred_domain = err.get("predicted_domain", "unknown")
        rule = err.get("rule_triggered", "unknown")
        analysis[true_domain][pred_domain][rule].append(err)

    for true_domain, pred_groups in analysis.items():
        domain_total_errors = sum(
            len(errs) for rules in pred_groups.values() for errs in rules.values()
        )
        print(f"\n[TRUE DOMAIN: {true_domain.upper()}] - {domain_total_errors} Errors")

        for pred_domain, rules in pred_groups.items():
            for rule, err_list in rules.items():
                print(
                    f"  -> Misclassified as {pred_domain.upper()} via '{rule}' ({len(err_list)} times)"
                )

                # Print up to 2 sample errors for context, including the scores
                for i, err in enumerate(err_list[:2]):
                    prompt = err["prompt"].replace("\n", " ")
                    display_prompt = prompt if len(prompt) < 85 else prompt[:82] + "..."

                    # Extract the score lines from the internal log
                    log_lines = err.get("router_internal_log", "").split("\n")
                    sem_info = next(
                        (line.strip() for line in log_lines if "SEMANTIC |" in line), ""
                    )
                    syn_info = next(
                        (line.strip() for line in log_lines if "SYNTAX   |" in line), ""
                    )

                    print(f"       Ex {i + 1}: {display_prompt}")
                    if sem_info:
                        print(f"              {sem_info.replace('-> ', '')}")
                    if syn_info:
                        print(f"              {syn_info.replace('-> ', '')}")
                print()
        print("-" * 90)


if __name__ == "__main__":
    main()
