import json
from pathlib import Path

import numpy as np
import pytest


def test_paired_summary_uses_seed_differences_and_is_json_safe():
    from aether.evidence.statistics import paired_summary

    result = paired_summary([103, 3, 53], [100, 0, 50], seed=9)
    assert result == {"n": 3, "mean": 3.0, "ci95": [3.0, 3.0]}
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("candidate,baseline", [([], []), ([1], [1, 2]),
    ([[1, 2]], [[1, 2]]), ([float("nan")], [1]), ([1], [float("inf")]),
    ([1 + 2j], [1]), ([True], [False]), (["1"], ["0"])])
def test_invalid_seed_samples_are_rejected(candidate, baseline):
    from aether.evidence.statistics import paired_summary

    with pytest.raises(ValueError):
        paired_summary(candidate, baseline)


def test_bootstrap_is_deterministic_and_reflects_seed_variability():
    from aether.evidence.statistics import paired_summary

    first = paired_summary([-4, -2, 0, 2, 4], [0] * 5, seed=12)
    assert first == paired_summary([-4, -2, 0, 2, 4], [0] * 5, seed=12)
    assert first["mean"] == 0
    assert first["ci95"][0] < 0 < first["ci95"][1]


def decision(n=30, **overrides):
    from aether.evidence.promotion import promotion_decision

    arguments = dict(retention_candidate=[0.9] * n, retention_baseline=[0.9] * n,
        retention_margin=0.02, replay_valid=True, integrity_valid=True,
        resources_comparable=True, min_confirmatory_seeds=30)
    arguments.update(overrides)
    return promotion_decision({"scalar": ([0.8] * n, [0.3] * n),
        "vector": ([0.8] * n, [0.5] * n)}, **arguments)


def test_positive_evidence_can_pass_all_explicit_gates():
    report = decision()
    assert report["promoted"] is True
    assert report["status"] == "promoted"
    assert report["reasons"] == []
    json.dumps(report, allow_nan=False)


def test_pilot_and_lowered_minimum_cannot_promote():
    assert decision(3)["status"] == "inconclusive"
    assert any("insufficient" in reason for reason in decision(3)["reasons"])
    with pytest.raises(ValueError):
        decision(3, min_confirmatory_seeds=3)


@pytest.mark.parametrize("gate", ["replay_valid", "integrity_valid", "resources_comparable"])
def test_failed_or_ambiguous_gates_reject_promotion(gate):
    for value in (False, None, "true", 1):
        report = decision(**{gate: value})
        assert not report["promoted"]
        assert any(gate in reason for reason in report["reasons"])


def test_retention_failure_and_one_unsuccessful_control_block_promotion():
    from aether.evidence.promotion import promotion_decision

    assert not decision(retention_candidate=[0.5] * 30)["promoted"]
    report = promotion_decision({"weak": ([1] * 30, [0] * 30),
        "strong": ([1] * 30, [1] * 30)}, retention_candidate=[1] * 30,
        retention_baseline=[1] * 30, retention_margin=0.01, replay_valid=True,
        integrity_valid=True, resources_comparable=True)
    assert not report["promoted"]
    assert any("strong" in reason for reason in report["reasons"])


def test_missing_controls_and_misaligned_retention_fail_closed():
    from aether.evidence.promotion import promotion_decision

    with pytest.raises(ValueError):
        decision(retention_candidate=[1] * 29)
    report = promotion_decision({}, retention_candidate=[1] * 30,
        retention_baseline=[1] * 30, retention_margin=0.01, replay_valid=True,
        integrity_valid=True, resources_comparable=True)
    assert not report["promoted"]


def test_receipt_detects_modified_missing_and_extra_artifacts(tmp_path):
    from aether.evidence.receipts import write_receipt, verify_receipt

    output = tmp_path / "run"
    output.mkdir()
    (output / "checkpoint.bin").write_bytes(b"checkpoint")
    (output / "results.json").write_text('{"return":0.5}', encoding="utf-8")
    receipt = write_receipt(output, {"study": "pilot"})
    assert isinstance(receipt, Path)
    assert verify_receipt(receipt)["valid"]
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert len(payload["source_fingerprint"]) == 64
    assert payload["environment"]["python"]
    assert len(payload["files"]) == 2
    with pytest.raises(FileExistsError):
        write_receipt(output)
    (output / "checkpoint.bin").write_bytes(b"altered")
    assert not verify_receipt(receipt)["valid"]
    (output / "checkpoint.bin").unlink()
    assert not verify_receipt(receipt)["valid"]
    (output / "extra.txt").write_text("extra")
    assert not verify_receipt(receipt)["valid"]


def test_receipts_reject_empty_and_nonfinite_content(tmp_path):
    from aether.evidence.receipts import write_receipt, verify_receipt

    with pytest.raises(ValueError):
        write_receipt(tmp_path)
    (tmp_path / "data.json").write_text('{"x":NaN}')
    with pytest.raises(ValueError):
        write_receipt(tmp_path)
    (tmp_path / "data.json").write_text('{"x":1}')
    with pytest.raises(ValueError):
        write_receipt(tmp_path, {"x": float("inf")})
    receipt = write_receipt(tmp_path)
    payload = json.loads(receipt.read_text())
    payload["files"] = []
    receipt.write_text(json.dumps(payload))
    assert not verify_receipt(receipt)["valid"]


def test_numeric_json_overflow_and_duplicate_keys_are_rejected(tmp_path):
    from aether.evidence.receipts import write_receipt

    artifact = tmp_path / "result.json"
    for content in ('{"value":1e400}', '{"value":1,"value":2}'):
        artifact.write_text(content)
        with pytest.raises(ValueError):
            write_receipt(tmp_path)
        assert not (tmp_path / "receipt.json").exists()


def test_multiple_failure_reasons_are_preserved():
    report = decision(3, replay_valid=False, integrity_valid=False,
                      resources_comparable=False, retention_candidate=[0.1] * 3)
    for required in ("replay_valid", "integrity_valid", "resources_comparable", "retention", "insufficient"):
        assert any(required in reason for reason in report["reasons"])


def test_verifier_rejects_missing_and_malformed_receipts(tmp_path):
    from aether.evidence.receipts import verify_receipt

    receipt = tmp_path / "receipt.json"
    assert not verify_receipt(receipt)["valid"]
    for content in ("[]", "{}", "null", '{"schema_version":1,"schema_version":2}',
                    '{"schema_version":1,"hash_algorithm":"sha256","files":1e400}'):
        receipt.write_text(content)
        assert not verify_receipt(receipt)["valid"]
    receipt.write_text('{"files":NaN}')
    assert not verify_receipt(receipt)["valid"]


@pytest.mark.parametrize("unsafe", ["../outside.txt", "/etc/passwd", "C:/Windows/system.ini",
    "nested/../../outside.txt", "..\\outside.txt", "//server/share/file"])
def test_receipt_verification_rejects_unsafe_references(tmp_path, unsafe):
    from aether.evidence.receipts import write_receipt, verify_receipt

    (tmp_path / "data.txt").write_text("data")
    receipt = write_receipt(tmp_path)
    payload = json.loads(receipt.read_text())
    payload["files"][0]["path"] = unsafe
    receipt.write_text(json.dumps(payload))
    result = verify_receipt(receipt)
    assert not result["valid"]
    assert result["details"]


def test_source_fingerprint_ignores_outputs_and_caches_but_tracks_source(tmp_path):
    from aether.evidence.receipts import source_fingerprint

    (tmp_path / "aether").mkdir()
    source = tmp_path / "aether" / "model.py"
    source.write_text("x=1")
    initial = source_fingerprint(tmp_path)
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "fake.py").write_text("ignored")
    (tmp_path / "aether" / "__pycache__").mkdir()
    (tmp_path / "aether" / "__pycache__" / "fake.py").write_text("ignored")
    assert source_fingerprint(tmp_path) == initial
    source.write_text("x=2")
    assert source_fingerprint(tmp_path) != initial


def test_symlink_escape_is_rejected_when_supported(tmp_path):
    from aether.evidence.receipts import write_receipt, verify_receipt

    output = tmp_path / "run"
    output.mkdir()
    (output / "data.txt").write_text("data")
    receipt = write_receipt(output)
    target = tmp_path / "outside.txt"
    target.write_text("data")
    (output / "data.txt").unlink()
    try:
        (output / "data.txt").symlink_to(target)
    except OSError:
        pytest.skip("Windows symlink creation requires privilege")
    assert not verify_receipt(receipt)["valid"]
