import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments.py"
SPEC = importlib.util.spec_from_file_location("qwc_experiments", MODULE_PATH)
experiments = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = experiments
SPEC.loader.exec_module(experiments)


class ExperimentStatisticsTest(unittest.TestCase):
    def test_bootstrap_mean_is_deterministic_and_contains_mean(self):
        first = experiments.bootstrap_mean_ci([9.0, 10.0, 11.0, 12.0], samples=2000, seed=7)
        second = experiments.bootstrap_mean_ci([9.0, 10.0, 11.0, 12.0], samples=2000, seed=7)
        self.assertEqual(first, second)
        mean, low, high = first
        self.assertEqual(mean, 10.5)
        self.assertLessEqual(low, mean)
        self.assertGreaterEqual(high, mean)

    def test_paired_ratio_uses_matching_trials(self):
        ratio = experiments.bootstrap_ratio_ci(
            {0: 10.0, 1: 20.0, 2: 30.0},
            {0: 20.0, 1: 40.0, 2: 60.0},
            samples=1000,
            seed=3,
        )
        self.assertIsNotNone(ratio)
        self.assertEqual(ratio, (2.0, 2.0, 2.0))

    def test_kv_ablation_has_one_well_formed_kv_option(self):
        scenario = next(
            item for item in experiments.suites()["ablations"]
            if item.name == "ablate-kv"
        )
        for variant in scenario.variants:
            command = experiments.command_for(scenario, variant)
            self.assertEqual(command.count("--kv-cache-dtype"), 1)

    def test_report_writes_intervals_and_paired_effect(self):
        records = [{"record_type": "experiment", "schema_version": 1}]
        for trial, base, tuned in ((0, 100.0, 120.0), (1, 102.0, 123.0), (2, 98.0, 118.0)):
            for index, (name, throughput) in enumerate((("base", base), ("tuned", tuned))):
                records.append({
                    "record_type": "measurement",
                    "status": "ok",
                    "scenario": "ablation",
                    "description": "fixture",
                    "variant": name,
                    "variant_index": index,
                    "trial": trial,
                    "result": {
                        "output_tokens_per_second": throughput,
                        "ttft_ms_p50": 10.0,
                        "itl_ms_p50": 5.0,
                    },
                })
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trials.jsonl"
            target = Path(directory) / "report.md"
            source.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            args = type("Args", (), {"input": source, "output": target, "seed": 11})()
            self.assertEqual(experiments.report(args), 0)
            text = target.read_text(encoding="utf-8")
            self.assertIn("95% CI", text)
            self.assertIn("base → tuned", text)


class RunHeaderPathTest(unittest.TestCase):
    def test_checkpoint_under_home_is_written_through_env_var(self):
        inside = Path.home().resolve() / "models/Qwen3.8-27B-QUASAR-NVFP4"
        self.assertEqual(
            experiments.display_path(inside),
            "$HOME/models/Qwen3.8-27B-QUASAR-NVFP4",
        )

    def test_checkpoint_outside_home_is_left_alone(self):
        # Скрывать там нечего, и путь должен остаться проверяемым.
        self.assertEqual(experiments.display_path(Path("/opt/ckpt/qwen")), "/opt/ckpt/qwen")


if __name__ == "__main__":
    unittest.main()
