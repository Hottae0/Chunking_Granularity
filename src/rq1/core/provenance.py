from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Provenance:
    fact_id: str
    document: str
    start: int
    end: int
    precision: str  # exact_quote or extraction_text_unit


def locate_extraction_units(document: str, source: str, units: list[dict]) -> dict[str, tuple[int, int]]:
    """Find official extraction-unit text in original source; fail on ambiguity.

    The official outputs supply text-unit IDs, not source character offsets.
    Sequential exact matching gives auditable coarse provenance.
    """
    spans, cursor = {}, 0
    for unit in units:
        text = str(unit["text"])
        start = source.find(text, cursor)
        if start < 0:
            start = source.find(text)
        if start < 0:
            raise ValueError(f"Cannot align extraction text unit {unit['id']} in {document}")
        spans[str(unit["id"])] = (start, start + len(text))
        cursor = start + 1
    return spans


def fact_provenance(fact_id: str, document: str, source: str,
                    unit_ids: list[str], unit_spans: dict[str, tuple[int, int]],
                    quote: str | None = None) -> list[Provenance]:
    records = []
    for unit_id in unit_ids:
        if unit_id not in unit_spans:
            continue
        start, end = unit_spans[unit_id]
        if quote:
            relative = source[start:end].find(quote)
            if relative >= 0:
                records.append(Provenance(fact_id, document, start + relative,
                                          start + relative + len(quote), "exact_quote"))
                continue
        records.append(Provenance(fact_id, document, start, end, "extraction_text_unit"))
    return records
