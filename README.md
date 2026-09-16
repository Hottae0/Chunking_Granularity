# Stage-specific chunk granularity — GraphRAG RQ1

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

`full.yaml`은 seed 42로 Novel 5편을 선택하고, 선택된 5편의 질문을 전부 사용해 64조건을 평가합니다.
실제 질문 수와 총 QA 호출 수는 입력 데이터에 따라 달라지며 manifest에 기록됩니다.
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
현재 실험 범위는 RQ1만 포함합니다. 소설 전체를 cluster 단위로 재표집하여 global optimum이
off-diagonal에 형성되는 빈도와 winner stability를 확인합니다. primary metric은 결과를 보기 전에 정하세요.

## 결과

| 파일 | 내용 |
| --- | --- |
| `per_query_results.csv` | 질문별 답변, QA/관계/근거 지표, latency, 실제 검색 토큰 |
| `config_summary.csv` | 64조건 평균과 오류 수 |
| `question_type_summary.csv` | 네 질문 유형별 조건 요약 |
| `rq1_analysis.json` | optimum 동률과 소설 단위 bootstrap winner 빈도 |
| `near_optimal_cells.csv` | 최대 F1에서 0.02 이내 영역 |
| `qa_heatmap.png` | Answer F1; 공식 평가 후 answer correctness로 갱신 |
| `relation_recall_heatmap.png` | 관계 recall 진단 지형 |
| `evidence_recall_heatmap.png` | 원문 span 기반 Evidence Recall@5 |
| `evidence_recall_proxy_heatmap.png` | 문장 단어 중첩 기반 Evidence Recall@5 proxy |
| `run_manifest.json`, `cache_identity.json` | 설정·데이터 식별·버전·완료 상태 |
| `cells/e*_r*/benchmark_predictions.json` | 공식 evaluator 입력 |

**Primary metric 열:** `official_answer_correctness`(공식 evaluator 실행 후), `qa_accuracy_proxy`,
`qa_em`, `answer_f1`. 공식 점수를 최종 주지표로 사용하고 로컬 EM/F1은 항상 저장합니다.

**Diagnostic metric 열:** `relation_recall_proxy`, `path_coverage_proxy`,
`evidence_recall_at_1/5/10`, `fixed_budget_evidence_recall`,
각 metric의 `*_proxy` 버전, `evidence_span_precision/recall/f1`.
모든 열은 `per_query_results.csv`와 cell 평균인 `config_summary.csv`에 저장됩니다.
정확한 evidence 문장을 원문에서 찾지 못한 span metric은 0이 아니라 빈 값(null)으로 남습니다.

`bootstrap_ci.json`은 기존 코드 호환용 **탐색적** 분석입니다. RQ1의 주 분석은
`rq1_analysis.json`에 기록됩니다. 누락 cell이나 실패/미채점 질문이 있으면 분석은 incomplete를
반환하고 성공 사례만 골라 최적값을 내지 않습니다.

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
python -m rq1.experiments.analysis runs/novel5_allq_8x8/per_query_results.csv \
  --metric official_answer_correctness
```

task에 따라 공식 evaluator가 answer_correctness를 제공하지 않을 수 있습니다. 이 경우 전체 혼합 분석은
incomplete이며 모든 task에 동일 지표가 있다고 가정하지 않습니다. 필요한 task subset을 사전 정의하세요.
judge, benchmark commit, 모델 버전도 실험 기록에 고정하세요.
공식 API usage는 직접 반환되지 않아 비용은 null일 수 있습니다. 서버 로그로 실제 비용을 확인하세요.

## 검증 범위

단위·합성 통합 테스트는 loader, provenance, 전체 grid, 캐시 무효화, wrong-source 평가,
소설 단위 bootstrap 안정성과 incomplete 처리 등을 검사합니다. mock은 lexical plumbing fixture이며 E에 따른
QA 효과를 검증하는 연구 결과가 아닙니다. 실제 모델을 사용한 전체 Novel 실험과 공식 judge는
서버 및 데이터 준비 후 별도로 실행해야 합니다.

## GPU가 할당된 서버·컨테이너에서 직접 실행

이미 GPU가 할당된 서버나 컨테이너 안에서 실행하는 방법입니다. 코드는 GPU를 추가로 할당하거나
`CUDA_VISIBLE_DEVICES`를 변경하지 않습니다. 생성 모델과 임베딩 모델은 vLLM의 OpenAI-compatible
endpoint로 실행하며, 기존 모델 서버가 있다면 `.env`에 해당 주소를 넣고 모델 실행 단계는 생략합니다.

### 준비

```bash
git clone https://github.com/Hottae0/Chunking_Granularity.git
cd Chunking_Granularity
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

데이터는 앞에서 설명한 `data/GraphRAG-Bench/` 구조에 둡니다. 코드·데이터·`runs/`는 컨테이너가
종료되어도 보존되는 디스크를 사용하세요. vLLM은 서버의 CUDA 환경과 호환되는 버전으로 별도 환경에
설치하는 것을 권장합니다. `.env`에는 실제 모델과 endpoint를 입력합니다.

```dotenv
BASE_URL=http://127.0.0.1:8000/v1
API_KEY=local-key
MODEL=Qwen/Qwen2.5-7B-Instruct
EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1
EMBEDDING_API_KEY=local-key
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSIONS=1024
LLM_BACKEND=openai_compatible
```

`EMBEDDING_DIMENSIONS`는 실제 임베딩 출력 차원과 같아야 합니다. 실행 전 점검에서 자동으로 비교합니다.

### 모델 서버와 실험 실행

현재 연구실 서버 지시에 따라 **GPU 0 한 장만 사용**합니다. 각 터미널에서
`export CUDA_VISIBLE_DEVICES="0"`을 먼저 실행하세요. 생성 모델은 GPU 메모리의 30%, 임베딩 모델은
6%를 기본 상한으로 사용해 현재 공유 GPU의 기존 작업과 공존할 여유를 둡니다. 서버 사용량이 바뀌면
관리자와 확인한 뒤 비율을 조정하세요. `VLLM_BIN`에는 설치된 vLLM 실행 파일 경로를 지정할 수 있습니다.

권장 실행은 한 명령으로 모델 서버의 시작과 종료까지 관리합니다.

```bash
export CUDA_VISIBLE_DEVICES="0"

# 작은 연결·저장 점검
bash scripts/run_server.sh configs/pilot.yaml

# Novel 5편, 선택된 소설의 질문 전부, 8×8 본 실험
bash scripts/run_server.sh configs/server.yaml
```

스크립트는 생성 모델과 임베딩 모델을 GPU 0에 올리고, endpoint/structured JSON/임베딩 차원을
검사한 뒤 실험을 실행합니다. 정상 종료, 오류, Ctrl+C 모두에서 스크립트가 자신이 시작한 vLLM
process group을 종료하므로 GPU 메모리가 반환됩니다. 모델 가중치 캐시는 서버 디스크에 남아 다음 실행의
다운로드를 줄입니다. 모델 로그는 결과 폴더의 `model_logs/`에 저장됩니다.

서버 저장소가 `/home/hottae0/Chunking_Granularity`에 있으면 본 실험 결과의 절대 경로는
`/home/hottae0/Chunking_Granularity/runs/server_novel5_allq_8x8`입니다. 완료 시
`results_complete.json`에 절대 출력 경로, 64개 cell 완료 수, 필수 결과 파일의 절대 경로가 기록됩니다.
중단 후 같은 명령을 다시 실행하면 완료된 그래프와 성공한 질문을 재사용합니다. 동일 output에 실험
process를 동시에 두 개 실행하지 마세요. 실제 속도는 공유 GPU의 다른 작업 부하에 영향을 받습니다.

## 연구실 서버 갱신·데이터·메모리 확인

서버의 기존 clone을 최신 `main`으로 갱신합니다.

```bash
cd /home/hottae0/Chunking_Granularity
git switch main
git pull --ff-only origin main
git rev-parse --short HEAD
```

GraphRAG-Bench 파일이 없으면 모델을 올리기 전에 다음을 실행합니다.

```bash
python -m pip install huggingface_hub
bash scripts/download_novel_data.sh
```

다운로드가 끝나면 다음 두 파일의 절대 경로와 Novel·질문 수가 출력됩니다.

```text
/home/hottae0/Chunking_Granularity/data/GraphRAG-Bench/Datasets/Corpus/novel.json
/home/hottae0/Chunking_Granularity/data/GraphRAG-Bench/Datasets/Questions/novel_questions.json
```

현재 H100의 81,559 MiB를 기준으로 vLLM 상한은 chat 30% 약 23.9 GiB,
embedding 6% 약 4.8 GiB로, 두 서버가 추가로 예약하는 상한은 약 28.7 GiB입니다.
기존 GPU 작업이 약 19–20 GiB라면 전체 사용량은 약 48–49 GiB 수준입니다.
Qwen2.5-7B BF16 가중치는 약 14–15 GiB, BGE-M3 가중치는 약 1–2 GiB이며
나머지는 KV cache와 실행 overhead입니다. 모델 다운로드·캐시에는 디스크 약 18–25 GiB를 예상합니다.
`CHAT_GPU_MEMORY_UTILIZATION`과 `EMBEDDING_GPU_MEMORY_UTILIZATION` 환경 변수로 상한을 조정할 수 있습니다.
5편·선택된 소설의 모든 질문·8×8 본 실험과 후속 분석은 모두 같은 Qwen2.5-7B 모델로 고정합니다. GPU 연산 사용률이 높은 시간에는
메모리가 남아도 실행 속도가 크게 느려질 수 있습니다.

