import json
from pathlib import Path

import pytest
import torch

from certainrag.signals.verifiers import Verifier

FIXTURE = Path(__file__).parent / "fixtures" / "parity.json"
TOLERANCE = 1e-3
FP16_TOLERANCE = 1e-2

pytestmark = pytest.mark.slow


def load_cases():
    if not FIXTURE.exists():
        return []
    fixture = json.loads(FIXTURE.read_text())
    claims = {p["id"]: p["claim"] for p in fixture["pairs"]}
    return [(name, row, claims[row["id"]]) for name, rows in fixture["models"].items() for row in rows]


CASES = load_cases()


@pytest.fixture(scope="module")
def verifiers():
    cache = {}

    def get(name, dtype=torch.float32):
        if (name, dtype) not in cache:
            cache[name, dtype] = Verifier(name, dtype=dtype)
        return cache[name, dtype]

    return get


@pytest.mark.skipif(not CASES, reason="tests/fixtures/parity.json not found")
@pytest.mark.parametrize("name, row, claim", CASES, ids=[f"{c[0]}-{c[1]['id']}" for c in CASES])
def test_matches_reference_implementation(verifiers, name, row, claim):
    probs = verifiers(name).score(row["chunks"], [claim] * len(row["chunks"]))
    assert probs == pytest.approx(row["probs"], abs=TOLERANCE)


@pytest.mark.skipif(not CASES or not torch.cuda.is_available(), reason="needs parity.json and a GPU")
@pytest.mark.parametrize("name, row, claim", CASES, ids=[f"{c[0]}-{c[1]['id']}" for c in CASES])
def test_fp16_stays_close_to_reference(verifiers, name, row, claim):
    probs = verifiers(name, torch.float16).score(row["chunks"], [claim] * len(row["chunks"]))
    assert probs == pytest.approx(row["probs"], abs=FP16_TOLERANCE)
