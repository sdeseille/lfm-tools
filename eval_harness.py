"""
Multi-intent tool-call detection eval harness.

Runs a fixed set of compound/multi-tool prompts against whatever model is
currently loaded behind the OpenAI-compatible server, N times each, and
records whether the expected tool set was detected — plus latency and
iteration count, since a "correct but spread over 3 round trips" result is
strictly worse than "correct in 1 round trip" for this comparison.

Results are appended (not overwritten) to a JSONL file, tagged with
--label, so you can restart the server with a different model/quant and
re-run this script to build up a comparable dataset across configs.

Usage:
    # Run against whatever model the server currently has loaded
    uv run eval_harness.py --label lfm2.5-350m-q8_0 --repeats 3

    # After running it against a few configs, compare them
    uv run eval_harness.py --compare

Requires the server to already be running (see main.py) and client_example.py
to be importable from the same directory.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

from client_example import OpenAIClient, ToolCallingAgent

DEFAULT_RESULTS_FILE = "eval_results.jsonl"

# Each case: a prompt plus the tool set it SHOULD trigger, regardless of
# phrasing, order, or how many sentences it's split across. Extend this list
# as you add tools or find new failure-prone phrasings.
TEST_CASES: List[Dict[str, Any]] = [
    {
        "id": "two_tool_shared_entity",
        "prompt": "What's the weather and the time in New York?",
        "expected_tools": {"get_weather", "get_time"},
    },
    {
        "id": "three_tool_single_connector",
        "prompt": "Calculate 4 * 3 and tell me the weather and time in New York.",
        "expected_tools": {"calculate", "get_weather", "get_time"},
    },
    {
        "id": "three_tool_separate_sentences",
        "prompt": "Calculate 4 * 3? What's the weather and the time in New York?",
        "expected_tools": {"calculate", "get_weather", "get_time"},
    },
    {
        "id": "calculate_nested_parens_plus_tool",
        "prompt": "What is (12+3)*2, and what's the weather in Paris?",
        "expected_tools": {"calculate", "get_weather"},
    },
    {
        "id": "three_tool_reordered",
        "prompt": "What's the time in Tokyo, the weather in Paris, and calculate 15 / 3.",
        "expected_tools": {"get_time", "get_weather", "calculate"},
    },
    {
        "id": "three_tool_french",
        "prompt": "Calcule 4 fois 3, et donne-moi la m\u00e9t\u00e9o et l'heure \u00e0 New York.",
        "expected_tools": {"calculate", "get_weather", "get_time"},
    },
]


def detected_tool_names(messages: List[Dict[str, Any]]) -> List[str]:
    """Every tool name the model actually called, across all iterations,
    in the order it called them (duplicates kept — a model calling the
    same tool twice is itself worth seeing, not just deduping away)."""
    names = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                names.append(tc["function"]["name"])
    return names


def iterations_until_all_tools_seen(messages: List[Dict[str, Any]]) -> int:
    """How many assistant turns it took before every tool call had been
    issued. 1 = the model got the whole compound request in one shot."""
    count = 0
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            count += 1
    return count


def run_case(agent: ToolCallingAgent, case: Dict[str, Any]) -> Dict[str, Any]:
    start = time.time()
    result = agent.run(case["prompt"], verbose=False)
    elapsed = time.time() - start

    detected = detected_tool_names(result["messages"])
    detected_set: Set[str] = set(detected)
    expected_set: Set[str] = case["expected_tools"]

    return {
        "case_id": case["id"],
        "prompt": case["prompt"],
        "expected_tools": sorted(expected_set),
        "detected_tools": detected,
        "missing_tools": sorted(expected_set - detected_set),
        "unexpected_tools": sorted(detected_set - expected_set),
        "passed": expected_set == detected_set,
        "iterations_to_gather_calls": iterations_until_all_tools_seen(result["messages"]),
        "total_iterations": result["iterations"],
        "status": result["status"],
        "elapsed_seconds": round(elapsed, 2),
        "llm_metrics": result.get("llm_metrics", []),
        "tool_metrics": result.get("tool_metrics", []),
    }


def run_suite(label: str, base_url: str, repeats: int, temperature: float,
              results_file: str) -> None:
    client = OpenAIClient(base_url=base_url)
    try:
        health = client.health_check()
    except Exception as e:
        print(f"⚠️  Couldn't reach server at {base_url}: {e}")
        print("    Make sure main.py is running first.")
        return
    print(f"✅ Connected — {health}\n")

    agent = ToolCallingAgent(client)
    agent.client.chat_completion.__defaults__  # noqa: keep default temp visible below
    records = []

    for case in TEST_CASES:
        print(f"── {case['id']} " + "─" * max(0, 60 - len(case['id'])))
        for run_idx in range(repeats):
            # temperature is applied inside chat_completion's default arg;
            # patch it per-call without touching client_example.py
            orig_chat_completion = agent.client.chat_completion

            def chat_with_temp(messages, model="LFM2.5-230M", tools=None,
                                tool_choice="auto", temperature=temperature,
                                max_tokens=256, _orig=orig_chat_completion):
                return _orig(messages, model=model, tools=tools,
                              tool_choice=tool_choice, temperature=temperature,
                              max_tokens=max_tokens)
            agent.client.chat_completion = chat_with_temp

            record = run_case(agent, case)
            agent.client.chat_completion = orig_chat_completion

            record["label"] = label
            record["run_index"] = run_idx
            record["temperature"] = temperature
            records.append(record)

            status = "✅ PASS" if record["passed"] else "❌ FAIL"
            print(f"   run {run_idx + 1}/{repeats}: {status}  "
                  f"detected={record['detected_tools']}  "
                  f"iters={record['iterations_to_gather_calls']}  "
                  f"{record['elapsed_seconds']}s")
            if not record["passed"]:
                if record["missing_tools"]:
                    print(f"      missing: {record['missing_tools']}")
                if record["unexpected_tools"]:
                    print(f"      unexpected: {record['unexpected_tools']}")
        print()

    with open(results_file, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    n_pass = sum(r["passed"] for r in records)
    print(f"Summary for '{label}': {n_pass}/{len(records)} runs passed "
          f"(appended to {results_file})")


def compare(results_file: str) -> None:
    path = Path(results_file)
    if not path.exists():
        print(f"No results file at {results_file} yet — run the suite first.")
        return

    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_label_case = defaultdict(list)
    for r in records:
        by_label_case[(r["label"], r["case_id"])].append(r)

    labels = sorted({r["label"] for r in records})
    case_ids = [c["id"] for c in TEST_CASES]

    header = f"{'case':<32}" + "".join(f"{lbl:<22}" for lbl in labels)
    print(header)
    print("-" * len(header))
    for cid in case_ids:
        row = f"{cid:<32}"
        for lbl in labels:
            runs = by_label_case.get((lbl, cid), [])
            if not runs:
                row += f"{'-':<22}"
                continue
            n_pass = sum(r["passed"] for r in runs)
            avg_latency = sum(r["elapsed_seconds"] for r in runs) / len(runs)
            avg_iters = sum(r["iterations_to_gather_calls"] for r in runs) / len(runs)
            row += f"{f'{n_pass}/{len(runs)} {avg_latency:.1f}s i={avg_iters:.1f}':<22}"
        print(row)

    print()
    for lbl in labels:
        lbl_records = [r for r in records if r["label"] == lbl]
        n_pass = sum(r["passed"] for r in lbl_records)
        avg_latency = sum(r["elapsed_seconds"] for r in lbl_records) / len(lbl_records)
        print(f"{lbl}: {n_pass}/{len(lbl_records)} overall, avg {avg_latency:.2f}s/call")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", help="Tag for this model/quant config, e.g. lfm2.5-350m-q8_0")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per test case")
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--results-file", default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--compare", action="store_true", help="Print comparison table across all labels seen so far, instead of running")
    args = parser.parse_args()

    if args.compare:
        compare(args.results_file)
        return

    if not args.label:
        parser.error("--label is required when running the suite (e.g. --label lfm2.5-350m-q8_0)")

    run_suite(args.label, args.base_url, args.repeats, args.temperature, args.results_file)


if __name__ == "__main__":
    main()
