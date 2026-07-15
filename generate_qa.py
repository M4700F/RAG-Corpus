"""
Generate 8 WH questions + answers per context entry from a Bengali
hallucination-detection dataset, using a local Ollama model.

Usage:
    python generate_qa.py --input samples.json --output questions.md --model qwen2.5:14b

Requires:
    pip install requests
"""

import json
import re
import argparse
import time
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

# Contexts to skip
NULL_MARKERS = {"[NULL]", "", None}

PROMPT_TEMPLATE = """তুমি একজন বাংলা ভাষা বিশেষজ্ঞ। নিচের প্যারাগ্রাফটি পড়ো এবং শুধুমাত্র এই প্যারাগ্রাফে উল্লেখিত তথ্যের ভিত্তিতে ৮টি "প্রশ্ন-উত্তর" জোড়া তৈরি করো।

নিয়মাবলি:
1. প্রতিটি প্রশ্ন হতে হবে WH (কী/কে/কবে/কোথায়/কেন/কীভাবে/কোনটি/কতজন ইত্যাদি) ধরনের।
2. প্রতিটি উত্তর অবশ্যই প্যারাগ্রাফে সরাসরি উল্লেখিত তথ্য থেকে আসতে হবে, নিজে থেকে কোনো তথ্য যোগ করবে না।
3. উত্তর যতটা সম্ভব সংক্ষিপ্ত রাখো (একটি শব্দ বা বাক্যাংশ), সম্পূর্ণ বাক্য নয়।
4. একই তথ্য নিয়ে দুইবার প্রশ্ন করো না, প্যারাগ্রাফের বিভিন্ন অংশ কভার করো।
5. আউটপুট অবশ্যই শুধুমাত্র বৈধ JSON array হতে হবে, অন্য কোনো ব্যাখ্যা বা টেক্সট থাকবে না।

আউটপুট ফরম্যাট (উদাহরণ):
[
  {{"question": "...", "answer": "..."}},
  ...
]

প্যারাগ্রাফ:
\"\"\"{context}\"\"\"
"""


def call_ollama(model, context, max_retries=3):
    prompt = PROMPT_TEMPLATE.format(context=context)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3},
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
            resp.raise_for_status()
            raw_text = resp.json().get("response", "").strip()
            return extract_json_array(raw_text)
        except Exception as e:
            print(f"  [retry {attempt+1}/{max_retries}] error: {e}")
            time.sleep(2)
    return None


def extract_json_array(text):
    """Pull out the first JSON array found in the model's output, tolerating
    stray preamble/markdown fences some models add despite instructions."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None

    candidate = text[start : end + 1]
    try:
        data = json.loads(candidate)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        return None
    return None


def normalize(s):
    """Loose normalization for substring grounding check (Bengali has no
    case, so this mainly strips punctuation/whitespace variance)."""
    return re.sub(r"[\s,।.()\"'\u200c\u200d]+", "", s or "")


def is_grounded(answer, context):
    return normalize(answer) in normalize(context)


def process_dataset(input_path, output_path, model, limit=None):
    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if limit:
        samples = samples[:limit]

    md_lines = ["# Generated WH Questions from Contexts\n"]
    skipped = 0
    total_generated = 0
    total_flagged = 0

    for idx, sample in enumerate(samples):
        context = sample.get("context")
        if context in NULL_MARKERS:
            skipped += 1
            continue

        print(f"[{idx+1}/{len(samples)}] Generating questions...")
        qa_pairs = call_ollama(model, context)

        md_lines.append(f"## Sample {idx+1}\n")
        md_lines.append(f"**Context:**\n\n> {context}\n")

        if not qa_pairs:
            md_lines.append("\n_⚠️ Generation failed for this sample._\n")
            print("  -> FAILED to generate/parse")
            continue

        md_lines.append("\n| # | Question | Answer | Grounded? |")
        md_lines.append("|---|----------|--------|-----------|")

        for i, qa in enumerate(qa_pairs[:8], 1):
            q = qa.get("question", "").strip()
            a = qa.get("answer", "").strip()
            grounded = is_grounded(a, context)
            total_generated += 1
            if not grounded:
                total_flagged += 1
            flag = "✅" if grounded else "❌ NOT FOUND IN CONTEXT"
            md_lines.append(f"| {i} | {q} | {a} | {flag} |")

        md_lines.append("\n---\n")

    md_lines.append(
        f"\n## Summary\n\n"
        f"- Samples processed: {len(samples) - skipped}\n"
        f"- Samples skipped (null context): {skipped}\n"
        f"- Total Q&A pairs generated: {total_generated}\n"
        f"- Flagged as NOT grounded (needs manual review): {total_flagged}\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nDone. Wrote {output_path}")
    print(f"Skipped {skipped} null-context samples.")
    print(f"Flagged {total_flagged}/{total_generated} answers as ungrounded — review these manually.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to samples.json")
    parser.add_argument("--output", default="questions.md", help="Output markdown file")
    parser.add_argument("--model", default="qwen2.5:14b", help="Ollama model name")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N samples (for testing)")
    args = parser.parse_args()

    process_dataset(args.input, args.output, args.model, args.limit)