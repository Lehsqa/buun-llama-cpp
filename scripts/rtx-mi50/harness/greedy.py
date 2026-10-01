#!/usr/bin/env python3
"""greedy.py --url URL --output F   |   greedy.py --compare A B
Fixed prompts at temperature 0, no prompt cache; used to check that a flag does not change greedy output."""
import argparse, json, sys, urllib.request

PROMPTS = [
    ("code", "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."),
    ("prose", "Explain in three short paragraphs why the sky is blue."),
    ("list", "List ten prime numbers greater than 100, one per line."),
    ("math", "What is 37 * 43? Show the multiplication steps."),
    ("long", "Summarize the following text in two sentences:\n" + ("The committee met on Tuesday to review the budget, "
             "discussed the delays in the bridge project, and agreed to publish a revised timeline next month. " * 60)),
]

def run(url, out):
    res = {"prompts": [p for p, _ in PROMPTS], "outputs": {}}
    for pid, text in PROMPTS:
        body = json.dumps({"messages": [{"role": "user", "content": text}], "max_tokens": 160, "temperature": 0,
                           "top_k": 1, "seed": 1, "cache_prompt": False,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=1800))
        res["outputs"][pid] = r["choices"][0]["message"].get("content") or ""
        print(pid, len(res["outputs"][pid]), "chars", flush=True)
    json.dump(res, open(out, "w"), indent=1)

def compare(a, b):
    A, B = json.load(open(a))["outputs"], json.load(open(b))["outputs"]
    same = 0
    for pid in A:
        x, y = A[pid], B.get(pid, "")
        if x == y:
            same += 1
        else:
            k = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))
            print("diff %s at char %d" % (pid, k))
    print("identical %d/%d" % (same, len(A)))
    return same == len(A)

ap = argparse.ArgumentParser()
ap.add_argument("--url"); ap.add_argument("--output"); ap.add_argument("--compare", nargs=2)
a = ap.parse_args()
if a.compare:
    sys.exit(0 if compare(*a.compare) else 1)
run(a.url, a.output)
