from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "code" / "scripts" / "validate_e1_prior_hierarchy_outputs.py"
ARMS = (
    "E1_prior_visible",
    "E1_prior_issuer_masked",
    "E1_prior_context_minimal",
    "E1_t1_evidence_visible_replay",
)


class ValidateE1PriorHierarchyOutputsTest(unittest.TestCase):
    def write_fixture(self, root: Path, *, replay_ids: list[str]) -> tuple[Path, Path, Path]:
        rendered = root / "rendered" / "E1_t1_evidence_visible_replay"
        rendered.mkdir(parents=True)
        (rendered / "evt_0001.json").write_text(
            json.dumps(
                {
                    "event_id": "evt_0001",
                    "treatment": "E1_t1_evidence_visible_replay",
                    "rendered_text": "Source evidence:\n- S001 [general]: text\nStructured facts:\n- X004 Fact: 1\n- X006 Fact: 2",
                }
            ),
            encoding="utf-8",
        )
        source = root / "rows.csv"
        fields = [
            "event_id", "treatment", "upstream_model_family", "profile", "profile_group",
            "model_family", "representation_seed", "decision_seed", "evidence_used",
        ]
        with source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for treatment in ARMS:
                writer.writerow(
                    {
                        "event_id": "evt_0001",
                        "treatment": treatment,
                        "upstream_model_family": "e1_prior_hierarchy",
                        "profile": "retail_day_trader",
                        "profile_group": "retail",
                        "model_family": "gpt-5.2",
                        "representation_seed": "0",
                        "decision_seed": "1",
                        "evidence_used": json.dumps(replay_ids if treatment == ARMS[-1] else []),
                    }
                )
        return source, root / "rendered", root / "report.json"

    def run_cli(self, source: Path, rendered: Path, report: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable, str(SCRIPT), "--input-csv", str(source),
                "--rendered-packets", str(rendered), "--report", str(report),
                "--expected-total", "4", "--expected-events", "1", "--expected-profiles", "1",
                "--expected-model-families", "gpt-5.2",
            ],
            cwd=PROJECT_ROOT, text=True, capture_output=True, check=False,
        )

    def test_accepts_source_and_structured_fact_ids_visible_in_replay_packet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source, rendered, report = self.write_fixture(Path(temporary_directory), replay_ids=["S001", "X004"])
            completed = self.run_cli(source, rendered, report)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(json.loads(report.read_text())["summary"]["all_valid"])

    def test_rejects_citation_not_visible_in_replay_packet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source, rendered, report = self.write_fixture(Path(temporary_directory), replay_ids=["S001", "X999"])
            completed = self.run_cli(source, rendered, report)
            self.assertEqual(completed.returncode, 1)
            errors = json.loads(report.read_text())["errors"]
            self.assertEqual(errors[0]["code"], "citation_not_visible_in_replay_packet")

    def test_rejects_citations_in_prior_only_arm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source, rendered, report = self.write_fixture(root, replay_ids=["S001"])
            with source.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["evidence_used"] = json.dumps(["S001"])
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            completed = self.run_cli(source, rendered, report)
            self.assertEqual(completed.returncode, 1)
            codes = {item["code"] for item in json.loads(report.read_text())["errors"]}
            self.assertIn("prior_arm_has_citations", codes)


if __name__ == "__main__":
    unittest.main()
