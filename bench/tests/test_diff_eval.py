import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "diff_eval.py"
SPEC = importlib.util.spec_from_file_location("qwc_diff_eval", SCRIPT)
diff_eval = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(diff_eval)


def step(token_id, scores):
    return {
        "token_id": token_id,
        "top_tokens": [
            {"token_id": score_id, "logprob": logprob, "rank": rank}
            for rank, (score_id, logprob) in enumerate(scores, 1)
        ],
    }


class DiffEvalTest(unittest.TestCase):
    def test_repeat_case_expands_to_exact_length(self):
        case = diff_eval.expand_case(
            {"id": "boundary", "repeat": {"token_ids": [2, 3], "length": 5}}
        )
        self.assertEqual(case["prompt_token_ids"], [2, 3, 2, 3, 2])

    def test_distribution_metrics_stop_at_first_divergence(self):
        reference = {
            "status": "ok",
            "prompt_token_ids": [9],
            "steps": [
                step(1, [(1, -0.1), (2, -1.0)]),
                step(2, [(2, -0.2), (3, -0.9)]),
                step(4, [(4, -0.1), (8, -2.0)]),
            ],
        }
        candidate = {
            "status": "ok",
            "prompt_token_ids": [9],
            "steps": [
                step(1, [(1, -0.2), (2, -1.1)]),
                step(7, [(7, -0.3), (3, -0.8)]),
                step(99, [(99, -0.1)]),
            ],
        }
        result = diff_eval.score_case(reference, candidate)
        self.assertEqual(result["exact_prefix"], 1)
        self.assertEqual(result["same_context_steps"], 2)
        self.assertEqual(result["top1_equal"], 1)
        self.assertEqual(result["topk_jaccard_count"], 2)

    def test_artifact_comparison_rejects_prompt_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            paths = []
            for name, prompt in (("reference", [1]), ("candidate", [2])):
                path = directory / f"{name}.jsonl"
                records = [
                    {
                        "record_type": "run",
                        "schema_version": 1,
                        "engine": name,
                        "engine_version": "test",
                    },
                    {
                        "record_type": "case",
                        "case_id": "case",
                        "prompt_token_ids": prompt,
                        "status": "ok",
                        "steps": [step(1, [(1, -0.1)])],
                    },
                ]
                path.write_text("".join(json.dumps(record) + "\n" for record in records))
                paths.append(path)
            score = diff_eval.score_artifact(paths[0], paths[1])
            self.assertEqual(score["cases"]["case"]["status"], "prompt_mismatch")


if __name__ == "__main__":
    unittest.main()
