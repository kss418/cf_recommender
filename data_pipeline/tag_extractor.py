import json
from pathlib import Path

input_path = Path("data/raw/codeforces_problems.json")
output_path = Path("data/processed/codeforces_tags.json")

data = json.loads(input_path.read_text(encoding="utf-8"))

problems = data["problems"]

tags = sorted({
    tag
    for problem in problems
    for tag in problem.get("tags", [])
})

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(
    json.dumps(tags, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"Saved {len(tags)} tags to {output_path}")