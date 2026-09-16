from __future__ import annotations

import json
import uuid
from pathlib import Path

from rq1.alignment.provenance import Provenance, align_fact, fact_provenance, locate_extraction_units
from rq1.chunking.fixed import fixed_chunks


def _ids(value) -> list[str]:
    if value is None:
        return []
    try:
        import pandas as pd
        if pd.isna(value) is True:
            return []
    except (ImportError, ValueError):
        pass
    return [str(x) for x in value]


def _uuid(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "rq1:" + value))


def adapt_view(graph_root: Path, documents, retrieval_size: int,
               overlap_ratio: float, cell_dir: Path, cached_chunks=None):
    """Replace extraction TextUnit links with retrieval-unit links by source span.

    Writes official-shaped parquet tables under a separate cell output directory.
    Keeps the extraction graph, reports and entity embedding store cached.
    """
    import pandas as pd
    graph_output = graph_root / "output"
    tables = {name: pd.read_parquet(graph_output / f"{name}.parquet") for name in
              ("entities", "relationships", "text_units", "documents", "communities", "community_reports")}
    titles = json.loads((graph_root / "input_titles.json").read_text(encoding="utf-8"))
    docs_by_name = {d.name: d for d in documents}
    doc_ids = {}
    for row in tables["documents"].to_dict("records"):
        title = Path(str(row["title"])).name
        if title not in titles:
            raise ValueError(f"Unknown official document title: {title}")
        doc_ids[str(row["id"])] = titles[title]
    units_by_doc: dict[str, list[dict]] = {d.name: [] for d in documents}
    for row in tables["text_units"].to_dict("records"):
        units_by_doc[doc_ids[str(row["document_id"])]].append(row)
    extraction_spans = {}
    for doc in documents:
        units = sorted(units_by_doc[doc.name], key=lambda r: int(r["human_readable_id"]))
        extraction_spans.update(locate_extraction_units(doc.name, doc.text, units))
    unit_lookup = {str(u["id"]): u for u in tables["text_units"].to_dict("records")}
    retrieval = []
    source_chunks = {}
    for doc in documents:
        chunks = ([c for c in cached_chunks if c.document == doc.name] if cached_chunks is not None
                  else fixed_chunks(doc.name, doc.text, retrieval_size, overlap_ratio))
        source_chunks[doc.name] = chunks
        retrieval.extend(chunks)
    original_to_retrieval: dict[str, set[str]] = {}
    for old_id, unit in unit_lookup.items():
        doc_name = doc_ids[str(unit["document_id"])]
        start, end = extraction_spans[old_id]
        coarse = Provenance(old_id, doc_name, start, end, "extraction_text_unit")
        original_to_retrieval[old_id] = set(align_fact(coarse, source_chunks[doc_name]))
    provenance: list[Provenance] = []
    mapped_entities: dict[str, set[str]] = {}
    mapped_relationships: dict[str, set[str]] = {}
    for table_name, mapped in (("entities", mapped_entities), ("relationships", mapped_relationships)):
        for row in tables[table_name].to_dict("records"):
            fact_id = str(row["id"])
            mapped[fact_id] = set()
            for unit_id in _ids(row.get("text_unit_ids")):
                extraction_unit = unit_lookup.get(unit_id)
                if extraction_unit is None:
                    continue
                doc_name = doc_ids[str(extraction_unit["document_id"])]
                doc = docs_by_name[doc_name]
                quote = str(row.get("title", "")) if table_name == "entities" else None
                spans = fact_provenance(fact_id, doc_name, doc.text, [unit_id], extraction_spans, quote)
                provenance.extend(spans)
                for span in spans:
                    ids = align_fact(span, source_chunks[doc_name])
                    mapped[fact_id].update(ids)
                    original_to_retrieval.setdefault(unit_id, set()).update(ids)
    new_ids = {c.id: _uuid(c.id) for c in retrieval}
    def remap(value):
        return sorted({new_ids[x] for old in _ids(value) for x in original_to_retrieval.get(old, ())})
    for table_name, mapped in (("entities", mapped_entities), ("relationships", mapped_relationships)):
        tables[table_name]["text_unit_ids"] = tables[table_name]["id"].apply(
            lambda ident: sorted(new_ids[x] for x in mapped[str(ident)]))
    for table_name in ("communities", "community_reports"):
        if "text_unit_ids" in tables[table_name]:
            tables[table_name]["text_unit_ids"] = tables[table_name]["text_unit_ids"].apply(remap)
    rows = []
    entity_inverse, relation_inverse = {}, {}
    for fact_id, chunk_ids in mapped_entities.items():
        for chunk_id in chunk_ids:
            entity_inverse.setdefault(chunk_id, []).append(fact_id)
    for fact_id, chunk_ids in mapped_relationships.items():
        for chunk_id in chunk_ids:
            relation_inverse.setdefault(chunk_id, []).append(fact_id)
    name_to_doc_id = {name: doc_id for doc_id, name in doc_ids.items()}
    for i, chunk in enumerate(retrieval):
        rows.append({"id": new_ids[chunk.id], "human_readable_id": i,
                     "text": chunk.text, "n_tokens": chunk.n_tokens,
                     "document_id": name_to_doc_id[chunk.document],
                     "entity_ids": sorted(entity_inverse.get(chunk.id, [])),
                     "relationship_ids": sorted(relation_inverse.get(chunk.id, [])),
                     "covariate_ids": []})
    tables["text_units"] = pd.DataFrame(rows)
    if "text_unit_ids" in tables["documents"]:
        tables["documents"]["text_unit_ids"] = tables["documents"]["id"].apply(
            lambda doc_id: [new_ids[c.id] for c in source_chunks[doc_ids[str(doc_id)]]])
    cell_output = cell_dir / "output"
    cell_output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_parquet(cell_output / f"{name}.parquet", index=False)
    provenance_frame = pd.DataFrame([p.__dict__ for p in provenance],
                                    columns=["fact_id", "document", "start", "end", "precision"])
    provenance_frame.to_parquet(cell_dir / "provenance.parquet", index=False)
    (cell_dir / "retrieval_spans.json").write_text(json.dumps([
        {"id": new_ids[c.id], "human_readable_id": i, "document": c.document,
         "start": c.start, "end": c.end} for i, c in enumerate(retrieval)], ensure_ascii=False), encoding="utf-8")
    return tables, retrieval, provenance
