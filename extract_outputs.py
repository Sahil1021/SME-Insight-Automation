import json
from pathlib import Path

output_folder = Path("outputs")

# Search all JSON files inside outputs AND its subfolders
for file in output_folder.rglob("*.json"):
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract the actual agent output
        if isinstance(data, dict):
            text = (
                data.get("insights")
                or data.get("recommendations")
                or data.get("output")
                or ""
            )

        elif isinstance(data, list):
            text = json.dumps(data, indent=2, ensure_ascii=False)

        else:
            text = str(data)

        # Keep the COMPLETE response exactly as stored
        output_file = file.with_suffix(".txt")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Created: {output_file}")

    except Exception as e:
        print(f"ERROR processing {file}: {e}")