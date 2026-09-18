"""Map source-offset provenance to independently chunked retrieval units."""
from rq1.core.provenance import Provenance, fact_provenance, locate_extraction_units
from rq1.core.types import Chunk


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def align_fact(fact: Provenance, retrieval_units: list[Chunk],
               minimum_overlap_ratio: float = 0.0) -> list[str]:
    result = []
    for unit in retrieval_units:
        if unit.document != fact.document:
            continue
        amount = overlap(fact.start, fact.end, unit.start, unit.end)
        if amount and amount / max(1, fact.end - fact.start) >= minimum_overlap_ratio:
            result.append(unit.id)
    return result


__all__ = ["Provenance", "align_fact", "fact_provenance", "locate_extraction_units", "overlap"]
