import os

# The exact broken string that was added (raw text)
OLD_HEADER = """Copyright (C) 2026 Özgür ÇOBAN

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

# The new, properly commented header
NEW_HEADER = f'"""\n{OLD_HEADER}"""\n\n'

TARGET_DIRS = ["src", "scripts"]


def fix_headers():
    for target_dir in TARGET_DIRS:
        if not os.path.exists(target_dir):
            print(f"Skipping {target_dir}: Directory not found.")
            continue

        for root, _, files in os.walk(target_dir):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)

                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()

                    # 1. REMOVE THE OLD BROKEN HEADER
                    # We use replace once to ensure we only hit the top instance
                    updated_content = content.replace(OLD_HEADER, "", 1)

                    # Clean up any leading whitespace left behind after removal
                    updated_content = updated_content.lstrip()

                    # 2. ADD THE NEW COMMENTED HEADER
                    # Avoid double-adding if you run this twice
                    if "Copyright (C) 2026 Özgür ÇOBAN" not in updated_content:
                        final_content = NEW_HEADER + updated_content

                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(final_content)
                        print(f"Fixed header in: {filepath}")
                    else:
                        # If the removal worked but the header was already there
                        # (e.g. manually fixed), just save the stripped version
                        if updated_content != content:
                            with open(filepath, "w", encoding="utf-8") as f:
                                f.write(updated_content)
                            print(f"Cleaned up broken text in: {filepath}")


if __name__ == "__main__":
    fix_headers()
    print("All files processed!")
