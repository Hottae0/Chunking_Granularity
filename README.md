# Stage-specific chunk granularity — GraphRAG RQ1 / RQ2

첨부 실험 설계의 fixed-length **8 × 8 factorial experiment** 구현입니다.
`g_E`(그래프 추출)와 `g_R`(근거 검색)를 각각
`128, 256, 512, 768, 1024, 1200, 1536, 2048` 토큰으로 바꿉니다.
PDF 표의 검색 열에는 2048이 빠져 있지만, 본문 8×8 설명에 맞춰 양쪽 모두 8개를 사용합니다.

## 재사용하는 것

- [Microsoft GraphRAG](https://github.com/microsoft/graphrag) 3.1.2의 standard indexing, entity embeddings, community reports, Local Search를 사용합니다. upstream을 복제해 고치지 않습니다.
- 기존 저장소의 chunking, provenance 정렬, API client, 공식 평가 연결, heatmap을 유지하고 오류를 수정했습니다.
- 전체 corpus의 그래프는 **E별 1회, 총 8회** 만들고 8개 R 조건에서 재사용합니다.
- 원문 검색 청크는 **R별 1회, 총 8세트** 만들고 각 E에서 재사용합니다.
- 성공한 질문은 재개 시 건너뜁니다. 데이터·질문·코드·모델·설정·주요 패키지 버전 fingerprint가 다르면 캐시 재사용을 거부합니다.
- 구버전 코드는 Git 이력에 남고, 현재 기본 브랜치는 이 구현을 사용합니다.

## 설치

Python 3.11 이상:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e .
cp .env.example .env      # Windows: Copy-Item .env.example .env
```

`.env`에 연구실 OpenAI-compatible 서버의 `BASE_URL`, `API_KEY`, `MODEL`,
`EMBEDDING_MODEL`을 설정합니다. GraphRAG가 사용하는 structured output 및 embedding을 지원해야 합니다.
첫 토크나이저 실행에는 공개 cl100k_base 파일 다운로드가 필요합니다.

[공식 GraphRAG-Bench](https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench)의 파일을 준비합니다:

```text
data/GraphRAG-Bench/Datasets/Corpus/novel.json
data/GraphRAG-Bench/Datasets/Questions/novel_questions.json
```

공식 `corpus_name/context`, 질문의 문자열 `evidence`, `evidence_triple` 필드를 읽습니다.
합성 예제의 문자열 배열 `evidence_relations`도 지원합니다.
질문 키는 `source::id`로 구분합니다. 실제 데이터의 소설 간 ID 충돌을 보존하며, 같은 소설 내 중복 ID, 없는 source, 빈 corpus는 오류로 처리합니다.

## 실행

```bash
python -m rq1.experiments.run_grid --config configs/synthetic.json
python -m unittest discover -s tests -v
python -m rq1.experiments.run_grid --config configs/pilot.yaml
python -m rq1.experiments.run_grid --config configs/full.yaml
```

`full.yaml`은 전체 20편/2,010개 질문 데이터 기준 64조건, **128,640 QA 호출**입니다.
실제 개수는 입력 데이터에 따라 달라지며 manifest에 기록됩니다.
`pilot.yaml`은 작은 사전 점검용입니다. 같은 명령으로 재개할 수 있습니다.
설정이나 코드를 바꾸면 새 `data.output`을 사용하세요. 구버전의 식별 정보 없는 캐시는 재사용하지 않습니다.

## 실험 통제와 해석

모든 조건에서 원문, 질문, 모델, tokenizer(cl100k_base), overlap(0), seed와 Local Search
context 상한(4096)을 고정합니다. text-unit 비중은 0.5, community 비중은 0입니다.
근거 진단 예산은 2048 토큰입니다. **같은 예산 상한이지, 실제 제공 토큰 수가 항상 같다는 뜻은 아닙니다.**
Local Search가 graph tables와 source rows를 함께 포장하므로 표 헤더와 fact 텍스트도 context를 차지합니다.
답변은 공식 Local Search의 `Single Sentence` 설정입니다. 장문 요약·창작 task의 native 설정과 다를 수 있으므로 task별 결과를 함께 보세요.

원문 문자 범위로 extraction unit과 retrieval unit을 연결합니다. fact→chunk와 chunk→fact는
동일한 fact-specific mapping을 사용합니다. 관계 provenance는 보통 extraction unit 전체로,
정밀한 mention-level evidence가 아닙니다. 따라서 관측 효과에는 provenance 폭의 효과도 포함됩니다.

RQ1은 전체 grid의 joint optimum, 동률, 0.02 이내 영역과 소설 단위 bootstrap 최적 영역 안정성을 봅니다.
개별 extraction fidelity의 독립 최적점을 증명한다고 해석하지 않습니다.
RQ2는 seed로 소설을 dev/test에 나눠 dev에서 best diagonal/off-diagonal을 선택하고,
선택한 두 조건을 고정하여 test 소설 단위 paired cluster bootstrap 95% CI를 계산합니다.
최소 4편이 필요하며 한 편 합성 예제에서는 RQ2가 unavailable인 것이 정상입니다.
원문 corpus는 공유하고 평가 질문을 소설별로 분리합니다. primary metric은 결과를 보기 전에 정하세요.

## 결과

| 파일 | 내용 |
| --- | --- |
| `per_query_results.csv` | 질문별 답변, QA/관계/근거 지표, latency, 실제 검색 토큰 |
| `config_summary.csv` | 64조건 평균과 오류 수 |
| `question_type_summary.csv` | 네 질문 유형별 조건 요약 |
| `rq1_rq2_analysis.json` | optimum 동률·bootstrap 빈도, dev 선택/test CI |
| `near_optimal_cells.csv` | 최대 F1에서 0.02 이내 영역 |
| `*_heatmap.png` | QA, relation, evidence 성능 지형 |
| `run_manifest.json`, `cache_identity.json` | 설정·데이터 식별·버전·완료 상태 |
| `cells/e*_r*/benchmark_predictions.json` | 공식 evaluator 입력 |

`bootstrap_ci.json`은 기존 코드 호환용 **탐색적** 분석입니다. 동일 데이터에서 선택한 최고점 비교이므로
논문 주장은 새 `rq1_rq2_analysis.json`의 held-out 결과를 사용하세요. 누락 cell이나 실패/미채점 질문이
있으면 새 분석은 incomplete를 반환하고 성공 사례만 골라 최적값을 내지 않습니다.

EM/F1/accuracy_proxy는 로컬 문자열 지표입니다. relation recall/path coverage와 evidence statement
recall은 단어 중첩 proxy입니다. evidence가 원문에 정확히 있으면 별도로 span 지표를 계산하고,
찾을 수 없으면 null입니다. 정확 근거 Recall@k의 hit는 문자 범위가 일부라도 겹치는 기준입니다.
근거 평가에서 잘못된 소설의 청크도 rank와 budget을 소비합니다. 공식 관계 정확도와 동일하지 않습니다.

## 공식 benchmark 평가

[공식 evaluator](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/tree/main/Evaluation)의 의존성과 judge를 준비한 뒤:

```bash
python -m rq1.experiments.official_eval --config configs/full.yaml \
  --benchmark-root /path/to/GraphRAG-Benchmark \
  --judge-model YOUR_JUDGE --embedding-model /path/to/bge-model
python -m rq1.experiments.analysis runs/novel20_8x8/per_query_results.csv \
  --metric official_answer_correctness
```

task에 따라 공식 evaluator가 answer_correctness를 제공하지 않을 수 있습니다. 이 경우 전체 혼합 분석은
incomplete이며 모든 task에 동일 지표가 있다고 가정하지 않습니다. 필요한 task subset을 사전 정의하세요.
judge, benchmark commit, 모델 버전도 실험 기록에 고정하세요.
공식 API usage는 직접 반환되지 않아 비용은 null일 수 있습니다. 서버 로그로 실제 비용을 확인하세요.

## 검증 범위

단위·합성 통합 테스트는 loader, provenance, 전체 grid, 캐시 무효화, wrong-source 평가,
소설별 검증 분리와 incomplete 처리 등을 검사합니다. mock은 lexical plumbing fixture이며 E에 따른
QA 효과를 검증하는 연구 결과가 아닙니다. 실제 모델을 사용한 전체 Novel 실험과 공식 judge는
서버 및 데이터 준비 후 별도로 실행해야 합니다.
