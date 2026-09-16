from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from rq1.alignment.provenance import Provenance, align_fact
from rq1.chunking.fixed import fixed_chunks
from rq1.config import load_settings
from rq1.data.graphrag_bench import load_novel
from rq1.eval.qa_metrics import qa_proxy
from rq1.eval.relation_metrics import relation_metrics
from rq1.eval.retrieval_metrics import evidence_metrics, evidence_statement_proxy
from rq1.experiments.bootstrap import paired_bootstrap
from rq1.experiments.cache import guard_run
from rq1.experiments.analysis import analyze
from rq1.llm.llm_client import make_client
from rq1.msgraphrag.byog_adapter import adapt_view
from rq1.msgraphrag.indexer import build_graph, graph_dir
from rq1.plotting.heatmaps import heatmap
from rq1.retrieval.local_search import MockLocalSearch, OfficialLocalSearch


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _mock_graph(documents, size, overlap):
    """Deterministic synthetic graph fixture. Official experiments use GraphRAG."""
    import re
    relations = []
    for doc in documents:
        for chunk in fixed_chunks(doc.name, doc.text, size, overlap):
            for match in re.finditer(r"(Alice|Bob)\s+(gave|returned|thanked)\s+(Alice|Bob)", chunk.text):
                relations.append({"id": f"{chunk.id}:{match.start()}",
                                  "source": match.group(1), "target": match.group(3),
                                  "description": match.group(0), "document": doc.name,
                                  "start": chunk.start + match.start(),
                                  "end": chunk.start + match.end()})
    return relations


def _mean(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if value not in (None, "", "None"):
            try:
                values.append(float(value))
            except ValueError:
                pass
    return sum(values) / len(values) if values else None


def _run_one(q, doc, search, chunks, relation_rows, settings, e, r, is_official):
    started = time.perf_counter()
    try:
        for attempt in range(settings.retries + 1):
            try:
                if is_official:
                    answer, ranked_ids, latency = search.query(q.question)
                    usage_in = usage_out = None  # GraphRAG LiteLLM usage lives in index/query logs.
                else:
                    answer, ranked_ids, latency, usage_in, usage_out = search.query(q.question, q.source)
                break
            except Exception:
                if attempt >= settings.retries:
                    raise
                time.sleep(settings.retry_delay_seconds * 2 ** attempt)
        selected = [chunks[i] for i in ranked_ids if 0 <= i < len(chunks)]
        qa = qa_proxy(answer, q.answer)
        evidence_exact = evidence_metrics(q.evidence, selected, q.source, doc.text, settings.evidence_tokens)
        evidence_proxy = evidence_statement_proxy(q.evidence, selected, settings.evidence_tokens)
        relation = relation_metrics(q.evidence_relations, relation_rows)
        result = {"id": q.id, "source": q.source, "question_type": q.question_type,
                  "g_E": e, "g_R": r, "setting": "shared" if e == r else "stage_specific",
                  "status": "ok", "question": q.question, "gold_answer": q.answer,
                  "generated_answer": answer, "query_latency_seconds": latency,
                  "wall_seconds": time.perf_counter() - started,
                  "retrieved_tokens": sum(c.n_tokens for c in selected),
                  "llm_input_tokens": usage_in, "llm_output_tokens": usage_out,
                  "reported_query_cost_usd": ((usage_in * settings.chat_input_usd_per_million
                                               + usage_out * settings.chat_output_usd_per_million) / 1_000_000)
                                               if usage_in is not None and usage_out is not None else None,
                  "cost_status": "reported_usage" if usage_in is not None else "usage_unavailable",
                  "retrieved_unit_ids": json.dumps([c.id for c in selected]),
                  "retrieval_context_parse_ok": bool(ranked_ids) if is_official else True}
        result.update(qa | evidence_exact | evidence_proxy | relation)
        benchmark = {"id": q.id, "question": q.question, "source": q.source,
                     "context": [c.text for c in selected], "evidence": list(q.evidence),
                     "question_type": q.question_type, "generated_answer": answer,
                     "gold_answer": q.answer, "ground_truth": q.answer}
        return result, benchmark
    except Exception as exc:
        return {"id": q.id, "source": q.source, "g_E": e, "g_R": r,
                "setting": "shared" if e == r else "stage_specific",
                "status": "error", "error": str(exc),
                "wall_seconds": time.perf_counter() - started}, None


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _manifest(settings, docs, questions, completed):
    return {"research_question": "Does the optimal chunk granularity differ between graph extraction and evidence retrieval?",
            "project_root": str(settings.project.resolve()),
            "output_directory": str(settings.output.resolve()),
            "result_files": ["per_query_results.csv", "config_summary.csv",
                             "question_type_summary.csv", "qa_heatmap.png",
                             "relation_recall_heatmap.png", "evidence_recall_heatmap.png",
                             "evidence_recall_proxy_heatmap.png", "rq1_analysis.json",
                             "near_optimal_cells.csv", "bootstrap_ci.json"],
            "primary_metrics": ["official_answer_correctness", "qa_accuracy_proxy", "qa_em", "answer_f1"],
            "diagnostic_metrics": ["relation_recall_proxy", "path_coverage_proxy",
                                   "evidence_recall_at_1", "evidence_recall_at_5",
                                   "evidence_recall_at_10", "fixed_budget_evidence_recall",
                                   "evidence_recall_at_1_proxy", "evidence_recall_at_5_proxy",
                                   "evidence_recall_at_10_proxy", "fixed_budget_evidence_recall_proxy",
                                   "evidence_span_precision", "evidence_span_recall",
                                   "evidence_span_f1"],
            "name": settings.name, "backend": settings.backend,
            "sizes": list(settings.sizes), "expected_graph_builds": len(settings.sizes),
            "expected_cells": len(settings.sizes) ** 2, "completed_cells": completed,
            "documents": [d.name for d in docs], "question_count": sum(len(x) for x in questions.values()),
            "document_selection": settings.document_selection,
            "overlap_ratio": settings.overlap_ratio,
            "context_tokens": settings.context_tokens, "evidence_tokens": settings.evidence_tokens,
            "seed": settings.seed, "base_url": settings.base_url, "model": settings.model,
            "embedding_model": settings.embedding_model, "embedding_base_url": settings.embedding_base_url,
            "chat_input_usd_per_million": settings.chat_input_usd_per_million,
            "chat_output_usd_per_million": settings.chat_output_usd_per_million,
            "cost_note": "Official GraphRAG API does not return token usage; query cost is null unless reported usage is available. Index logs preserve server-side accounting evidence.",
            "versions": {name: _version(name) for name in ("graphrag", "pandas", "tiktoken")},
            "corpus_sha256": hashlib.sha256("".join(d.text for d in docs).encode()).hexdigest(),
            "evaluation_note": "Local EM/F1 are transparent proxies; official benchmark judge scores require the official evaluator."}


def _summarize(settings, rows):
    summaries = []
    metrics = ("qa_em", "answer_f1", "qa_accuracy_proxy", "relation_recall_proxy",
               "path_coverage_proxy", "evidence_recall_at_1", "evidence_recall_at_5",
               "evidence_recall_at_10", "fixed_budget_evidence_recall",
               "evidence_recall_at_1_proxy", "evidence_recall_at_5_proxy",
               "evidence_recall_at_10_proxy", "fixed_budget_evidence_recall_proxy",
               "evidence_span_precision",
               "evidence_span_recall", "evidence_span_f1", "query_latency_seconds", "retrieved_tokens",
               "reported_query_cost_usd")
    for e in settings.sizes:
        for r in settings.sizes:
            group = [x for x in rows if int(x["g_E"]) == e and int(x["g_R"]) == r]
            valid = [x for x in group if x.get("status") == "ok"]
            result = {"g_E": e, "g_R": r, "setting": "shared" if e == r else "stage_specific",
                      "n_ok": len(valid), "n_error": len(group) - len(valid)}
            result.update({metric: _mean(valid, metric) for metric in metrics})
            summaries.append(result)
    return summaries


def _optimal_region(summaries, metric="answer_f1", delta=0.02):
    valid = [x for x in summaries if x[metric] is not None and x["n_ok"] > 0]
    if not valid:
        return [], {"metric": metric, "threshold": None, "cells": 0}
    best = max(float(x[metric]) for x in valid)
    near = [x for x in valid if float(x[metric]) >= best - delta]
    return near, {"metric": metric, "best": best, "within_absolute_delta": delta,
                  "threshold": best - delta, "cells": len(near),
                  "shared_cells": sum(x["setting"] == "shared" for x in near),
                  "stage_specific_cells": sum(x["setting"] == "stage_specific" for x in near),
                  "extraction_sizes": sorted({x["g_E"] for x in near}),
                  "retrieval_sizes": sorted({x["g_R"] for x in near})}


def run(config_path: Path) -> Path:
    settings = load_settings(config_path)
    if settings.backend == "official":
        os.environ["GRAPHRAG_API_KEY"] = settings.api_key
        os.environ["GRAPHRAG_EMBEDDING_API_KEY"] = settings.embedding_api_key
    settings.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True,
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(settings.output / "run.log", encoding="utf-8")])
    logger = logging.getLogger(__name__)
    docs, questions = load_novel(settings.corpus, settings.questions,
                                 settings.max_documents, settings.max_questions_per_document,
                                 settings.document_selection, settings.seed)
    guard_run(settings, docs, questions)
    doc_lookup = {d.name: d for d in docs}
    all_questions = [q for d in docs for q in questions[d.name]]
    for e in settings.sizes:
        if settings.backend == "official":
            build_graph(settings, docs, e, logger)
        else:
            root = graph_dir(settings.output, e)
            root.mkdir(parents=True, exist_ok=True)
            file = root / "mock_relations.json"
            if not file.exists():
                file.write_text(json.dumps(_mock_graph(docs, e, settings.overlap_ratio)), encoding="utf-8")
    retrieval_cache = {r: [c for d in docs for c in fixed_chunks(d.name, d.text, r, settings.overlap_ratio)]
                       for r in settings.sizes}
    for e in settings.sizes:
        root = graph_dir(settings.output, e)
        for r in settings.sizes:
            cell_dir = settings.output / "cells" / f"e{e}_r{r}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cell_dir / "per_query.csv"
            cached = {str(x["id"]): x for x in _read_csv(cache_path) if x.get("status") == "ok"}
            pending = [q for q in all_questions if q.id not in cached]
            if not pending and (cell_dir / "benchmark_predictions.json").exists():
                logger.info("Reuse cell e%d r%d", e, r)
                continue
            logger.info("Cell e%d r%d: %d pending queries", e, r, len(pending))
            if settings.backend == "official":
                tables, chunks, _ = adapt_view(root, docs, r, settings.overlap_ratio, cell_dir, retrieval_cache[r])
                search = OfficialLocalSearch(root, tables, settings.community_level)
                relation_rows = tables["relationships"].to_dict("records")
            else:
                client = make_client(settings)
                chunks = retrieval_cache[r]
                graph = json.loads((root / "mock_relations.json").read_text(encoding="utf-8"))
                for fact in graph:
                    span = Provenance(fact["id"], fact["document"], fact["start"], fact["end"], "exact_quote")
                    fact["retrieval_unit_ids"] = align_fact(span, chunks)
                (cell_dir / "mock_alignment.json").write_text(json.dumps(graph), encoding="utf-8")
                search = MockLocalSearch(chunks, client, settings.evidence_tokens)
                relation_rows = graph
            benchmarks = []
            with ThreadPoolExecutor(max_workers=settings.workers) as pool:
                futures = {pool.submit(_run_one, q, doc_lookup[q.source], search, chunks,
                                       relation_rows, settings, e, r, settings.backend == "official"): q
                           for q in pending}
                for future in as_completed(futures):
                    result, benchmark = future.result()
                    cached[result["id"]] = result
                    if benchmark:
                        benchmarks.append(benchmark)
                    _write_csv(cache_path, list(cached.values()))
            # Export the exact official benchmark prediction schema for its judge.
            exported = []
            for q in all_questions:
                row = cached[q.id]
                if row.get("status") == "ok":
                    ids = json.loads(row["retrieved_unit_ids"])
                    unit_lookup = {c.id: c.text for c in chunks}
                    exported.append({"id": q.id, "question": q.question, "source": q.source,
                                     "context": [unit_lookup[x] for x in ids if x in unit_lookup],
                                     "evidence": list(q.evidence), "question_type": q.question_type,
                                     "generated_answer": row["generated_answer"],
                                     "gold_answer": q.answer, "ground_truth": q.answer})
            (cell_dir / "benchmark_predictions.json").write_text(
                json.dumps(exported, ensure_ascii=False, indent=2), encoding="utf-8")
    all_rows = [row for file in sorted((settings.output / "cells").glob("e*_r*/per_query.csv"))
                for row in _read_csv(file)]
    _write_csv(settings.output / "per_query_results.csv", all_rows)
    summaries = _summarize(settings, all_rows)
    _write_csv(settings.output / "config_summary.csv", summaries)
    heatmap(summaries, settings.sizes, "answer_f1", settings.output / "qa_heatmap.png", "QA Answer F1 (local proxy)")
    heatmap(summaries, settings.sizes, "relation_recall_proxy", settings.output / "relation_recall_heatmap.png",
            "Query-relevant relation recall (lexical proxy)")
    heatmap(summaries, settings.sizes, "evidence_recall_at_5", settings.output / "evidence_recall_heatmap.png",
            "Evidence Recall@5 (source-span overlap)")
    heatmap(summaries, settings.sizes, "evidence_recall_at_5_proxy",
            settings.output / "evidence_recall_proxy_heatmap.png",
            "Evidence statement Recall@5 (lexical proxy)")
    near, region = _optimal_region(summaries)
    _write_csv(settings.output / "near_optimal_cells.csv", near)
    (settings.output / "optimal_region_summary.json").write_text(json.dumps(region, indent=2), encoding="utf-8")
    try:
        bootstrap = paired_bootstrap(all_rows, seed=settings.seed)
        (settings.output / "bootstrap_ci.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    except ValueError as exc:
        logger.warning("Bootstrap unavailable: %s", exc)
    analysis = analyze(all_rows, sizes=settings.sizes, seed=settings.seed)
    (settings.output / "rq1_analysis.json").write_text(json.dumps(analysis, indent=2))
    type_rows = []
    for kind in sorted({x.get("question_type", "unknown") for x in all_rows}):
        type_rows.extend(dict(row, question_type=kind) for row in
                         _summarize(settings, [x for x in all_rows if x.get("question_type") == kind]))
    _write_csv(settings.output / "question_type_summary.csv", type_rows)
    completed = sum(len(rows := _read_csv(settings.output / "cells" / f"e{e}_r{r}" / "per_query.csv")) == len(all_questions)
                    and all(x.get("status") == "ok" for x in rows)
                    for e in settings.sizes for r in settings.sizes)
    (settings.output / "run_manifest.json").write_text(
        json.dumps(_manifest(settings, docs, questions, completed), ensure_ascii=False, indent=2), encoding="utf-8")
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)
        handler.close()
    return settings.output


def main():
    parser = argparse.ArgumentParser(description="MS GraphRAG dual segmentation factorial grid")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
