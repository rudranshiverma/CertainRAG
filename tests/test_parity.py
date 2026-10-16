import json
from pathlib import Path

import pytest

from certainrag.signals.verifiers import Verifier

FIXTURE = Path(__file__).parent / "fixtures" / "parity.json"
TOLERANCE = 1e-3

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

    def get(name):
        if name not in cache:
            cache[name] = Verifier(name)
        return cache[name]

    return get


@pytest.mark.skipif(not CASES, reason="tests/fixtures/parity.json not found")
@pytest.mark.parametrize("name, row, claim", CASES, ids=[f"{c[0]}-{c[1]['id']}" for c in CASES])
def test_matches_reference_implementation(verifiers, name, row, claim):
    probs = verifiers(name).score(row["chunks"], [claim] * len(row["chunks"]))
    assert probs == pytest.approx(row["probs"], abs=TOLERANCE)
