# GraphRAG 청크 크기 실험 (RQ1)

**연구 질문:** 그래프 추출에 좋은 청크 크기와 근거 검색에 좋은 청크 크기는 다른가?

MS GraphRAG **Local Search**에서 두 크기를 독립적으로 바꿔 봅니다.

| 기호 | 단계 | 크기 후보 |
| --- | --- | --- |
| `g_E` | entity/relationship 추출 | 128, 256, 512, 768, 1024, 1200, 1536, 2048 |
| `g_R` | 답변 근거가 되는 원문 검색 | 같은 8개 값 |

두 값이 같은 대각선 8개 cell은 **shared-chunk baseline**, 다른 56개 cell은 **stage-specific 설정**입니다. 기본 실행은 GraphRAG-Bench **Novel 5편을 seed 42로 선택하고, 해당 5편의 질문은 전부 사용**합니다.

> 연구실 서버 주소와 모델명은 나중에 `.env`에 넣으면 됩니다. 서버 없이 가능한 mock 테스트는 맨 아래에 있습니다.

## 1. 설치

Python 3.11 이상에서 프로젝트 폴더를 열고 다음을 실행합니다.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .env.example .env
```

[GraphRAG-Bench 공식 dataset](https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench)에서 데이터를 받아 아래에 둡니다.

```text
data/GraphRAG-Bench/
└─ Datasets/
   ├─ Corpus/novel.json
   └─ Questions/novel_questions.json
```

실험에 사용한 dataset 버전 또는 commit도 기록하세요. 원문이나 질문 파일이 바뀌면 이전 실행 캐시를 다시 쓰면 안 됩니다.

## 2. 연구실 LLM 서버 설정

`.env`의 네 값을 연구실 서버 정보로 바꿉니다.

```dotenv
BASE_URL=http://your-lab-server/v1
API_KEY=your-key
MODEL=your-chat-model
EMBEDDING_MODEL=your-embedding-model
```

`BASE_URL`은 OpenAI-compatible API의 `/v1` 주소입니다. 공식 MS GraphRAG index와 Local Search는 내부 LiteLLM으로 completion/embedding API를 호출합니다. 서버는 graph 추출용 structured JSON output과 embedding 호출을 지원해야 합니다.

연구실 서버가 OpenAI-compatible하지 않다면 `src/rq1/llm/base.py`의 `LLMClient`를 구현할 수 있습니다. **공식 GraphRAG 경로까지 비호환 서버로 실행하려면** GraphRAG의 [custom model protocol](https://microsoft.github.io/graphrag/config/models/)도 등록해야 합니다. `mock` backend는 동작 점검용이며 연구 성능으로 해석하지 않습니다.

## 3. 8×8 실행

```powershell
python -m rq1.experiments.run_grid --config configs/full.yaml
```

`configs/full.yaml`의 기본값은 다음과 같습니다.

| 설정 | 값 |
| --- | --- |
| Novel | seed 42로 선택한 5편 |
| 질문 | 선택한 5편의 질문 전부 |
| `g_E × g_R` | 8 × 8 = 64 cell |
| chunk overlap | 0 |
| Local Search context budget | 모든 cell에서 4096 token |
| evidence 진단 budget | 모든 cell에서 2048 token |

선택된 Novel 제목과 질문 수는 실행 후 `run_manifest.json`에서 확인할 수 있습니다. 먼저 작은 동작 확인이 필요하면 `configs/pilot.yaml`로 4×4를 실행할 수 있고, 나중에 Novel 전체로 확장하려면 `configs/all_novels.yaml`을 사용합니다.

```powershell
python -m rq1.experiments.run_grid --config configs/pilot.yaml
python -m rq1.experiments.run_grid --config configs/all_novels.yaml
```

Windows용 `scripts/run_full.ps1`/`run_pilot.ps1`와 Linux용 `.sh`도 있습니다. 중단된 실행은 같은 명령으로 재개할 수 있습니다. **진행 중인 run의 모델, corpus, overlap, token budget은 바꾸지 마세요.** 설정을 바꿀 때는 output 경로도 새로 지정해야 캐시가 섞이지 않습니다.

## 그래프를 8회만 만드는 방법

MS GraphRAG 기본 파이프라인은 추출과 검색에 하나의 `TextUnit` 분할을 공유합니다. 이 실험에서는:

1. 공식 standard indexer로 `g_E`별 그래프를 **한 번씩**, 총 8회 만듭니다.
2. 각 그래프의 공식 entities/relationships, community tables, embedding store를 재사용합니다.
3. 같은 원문을 `g_R`로 다시 나누어 검색용 `text_units.parquet`를 만듭니다.
4. 추출 unit과 검색 unit이 겹치는 **원문 문자 범위**로 entity/relationship 링크를 다시 매핑합니다.
5. 각 `(g_E, g_R)`의 parquet, 결과, 재개 캐시는 별도 `cells/e*_r*/` 폴더에 둡니다.

따라서 **64번 재-indexing하지 않습니다.** 기존 추출 `text_unit_ids`를 검색 unit ID로 그대로 복사하지도 않습니다. graph fact와 원문 청크를 함께 쓰는 **Local Search**를 선택한 이유도 여기에 있습니다. Global Search는 community report 중심이라 `g_R` 효과를 직접 보기 어렵습니다.

**Provenance 한계:** 공식 GraphRAG 관계 출력은 관계를 뒷받침하는 정확한 원문 quote/offset을 주지 않습니다. entity title이 원문에 있으면 그 위치를 저장하지만, 관계는 보통 *추출 TextUnit 전체 범위*를 저장합니다. 이 범위가 여러 검색 unit에 관계를 연결할 수 있으므로 `provenance.parquet`의 `precision`을 확인해야 합니다. 추출 TextUnit을 원문에서 정확히 찾지 못하면 실행을 중단합니다.

## 4. 결과 확인

기본 8×8 결과는 `runs/novel5_8x8/` 아래에 저장됩니다.

| 파일 | 내용 |
| --- | --- |
| `per_query_results.csv` | 질문 × cell별 답변, 지표, latency, token |
| `config_summary.csv` | cell별 평균과 성공/실패 질문 수 |
| `qa_heatmap.png` | QA 성능 지형 |
| `relation_recall_heatmap.png` | 관계 진단 지형 |
| `evidence_recall_heatmap.png` | 근거 검색 진단 지형 |
| `run_manifest.json` | Novel, 설정, 버전, corpus hash |
| `bootstrap_ci.json` | 최고 대각선과 비대각선의 paired bootstrap CI |
| `near_optimal_cells.csv` | 최고 F1에서 0.02 이내인 cell |

**Primary downstream metric**은 QA입니다. 기본 grid의 `qa_em`/`answer_f1`은 공개된 정규화와 토큰 중첩 방식으로 계산합니다. `qa_accuracy_proxy`는 EM과 같은 정확 일치 비율이며 **공식 benchmark Accuracy가 아닙니다**.

GraphRAG-Bench는 task type별 `answer_correctness`, ROUGE, coverage 등을 제공합니다. 최종 비교에는 [공식 generation evaluator](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/blob/main/Evaluation/generation_eval.py)의 `answer_correctness`를 우선 사용하세요. evaluator repository와 judge LLM/BGE embedding을 준비한 뒤 실행합니다.

```powershell
python -m rq1.experiments.official_eval `
  --config configs/full.yaml `
  --benchmark-root C:\path\to\GraphRAG-Benchmark `
  --judge-model your-judge-model `
  --embedding-model C:\path\to\bge-model
```

이 명령은 cell별 공식 형식 prediction을 평가하고 `official_answer_correctness`를 query/summary에 추가하며 QA heatmap을 공식 점수로 다시 그립니다. evaluator 의존성은 해당 repository 지침에 따라 설치해야 합니다.

**Diagnostic metric**은 QA 차이의 원인을 살피는 데 사용합니다. 관계 recall/path coverage와 evidence Recall@1/@5/@10 및 고정 token 예산 recall은 gold annotation과 출력 텍스트의 **단어 중첩 proxy**입니다. gold evidence 문장이 원문에 정확히 있을 때만 문자 범위 기반 evidence recall과 span precision/recall/F1도 계산합니다. 일치하는 원문 범위가 없으면 `null`이며 0으로 평균하지 않습니다. 이 진단 점수만으로 최적 설정을 결정하지 마세요.

## 5. 통계와 재현성

```powershell
python -m rq1.experiments.bootstrap runs/novel5_8x8/per_query_results.csv `
  --metric answer_f1
```

스크립트는 **같은 질문**에서 최고 shared cell과 최고 stage-specific cell을 비교하고 paired bootstrap CI를 계산합니다. `near_optimal_cells.csv`도 함께 보세요. 한 최고점만 튀는지, 높은 성능이 인접한 설정에도 유지되는지가 연구 질문에 중요합니다.

같은 질문으로 최적 cell을 고르고 CI까지 계산하면 **선택 편향**이 있습니다. 결론은 dev에서 설정을 고른 뒤 held-out Novel/질문에서 다시 비교하는 방식으로 확인해야 합니다. 모델, prompt, tokenizer, embedding 모델, judge, clustering seed, 데이터 버전, budget, overlap도 고정하세요.

질의 latency와 retrieved token은 각 질문에 기록합니다. 공식 GraphRAG API는 LLM usage를 질의 결과에 직접 반환하지 않아 실제 query 비용은 `null`이며, 연구실 서버 usage 로그와 `index.log`/`query.log`를 함께 확인해야 합니다. MS GraphRAG 버전을 고정하세요. 이 프로젝트는 **공식 standard index와 공식 Local Search API**를 쓰지만, dual segmentation과 위치 정렬은 연구용 custom layer입니다.

## 서버 없이 동작 확인

```powershell
python -m unittest discover -s tests -v
python -m rq1.experiments.run_grid --config configs/synthetic.json
```

작은 synthetic 문서를 mock LLM으로 돌려 파일 생성과 재개 경로를 확인합니다. 이 점수는 실제 Novel benchmark 결과가 아닙니다.
