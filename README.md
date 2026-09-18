# Stage-specific chunk granularity — GraphRAG RQ1

첨부 실험 설계의 fixed-length chunk granularity 실험 구현입니다. 같은 코드에서 설정 파일만 바꿔 두 모드로 실행합니다.

- **사전실험 3×3:** `g_E, g_R ∈ [256, 512, 1024]`
- **본 실험 8×8:** `g_E, g_R ∈ [128, 256, 512, 768, 1024, 1200, 1536, 2048]`

사전실험은 빠르게 경향을 확인하는 screening 용도이며, 최종 RQ1 결론은 본 실험과 공식 평가를 기준으로 냅니다.

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
`pilot.yaml`은 작은 연결 점검용입니다. 연구용 사전실험은 `server_preliminary_3x3.yaml`,
본 실험은 `server.yaml`을 사용합니다. 두 설정은 같은 5편과 전체 질문을 사용하고 결과 폴더만 분리됩니다.
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
| `config_summary.csv` | 사전실험 9조건 또는 본 실험 64조건의 평균과 오류 수 |
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

이미 GPU가 할당된 서버나 컨테이너 안에서 실행하는 방법입니다. 일반 Python 모듈은 GPU를 추가로 할당하지 않습니다. 실행 전에 터미널에서
`CUDA_VISIBLE_DEVICES`를 지정하며 서버 실행 스크립트는 그 값을 그대로 사용합니다. 생성 모델과 임베딩 모델은 vLLM의 OpenAI-compatible
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

사용할 물리 GPU는 실행 터미널에서 명시적으로 선택합니다. `run_server.sh`는
`CUDA_VISIBLE_DEVICES`를 덮어쓰지 않으며 값이 설정되지 않았으면 실행을 중단합니다. 생성 모델은 GPU 메모리의 30%, 임베딩 모델은
6%를 기본 상한으로 사용해 현재 공유 GPU의 기존 작업과 공존할 여유를 둡니다. 서버 사용량이 바뀌면
관리자와 확인한 뒤 비율을 조정하세요. `VLLM_BIN`에는 설치된 vLLM 실행 파일 경로를 지정할 수 있습니다.

권장 실행은 한 명령으로 모델 서버의 시작과 종료까지 관리합니다.

```bash
export CUDA_VISIBLE_DEVICES="1"

# 작은 연결·저장 점검
bash scripts/run_server.sh configs/pilot.yaml

# 사전실험 준비: 중앙 graph store의 기존 그래프 검증
bash scripts/prepare_preliminary_3x3.sh

# Novel 5편, 선택된 소설의 질문 전부, 3×3 사전실험
bash scripts/run_server.sh configs/server_preliminary_3x3.yaml

# Novel 5편, 선택된 소설의 질문 전부, 8×8 본 실험
bash scripts/run_server.sh configs/server.yaml
```

### E512만 인덱싱

기존 소설 5편(seed 42)을 유지하면서 중앙 graph store의 `e512`만 인덱싱합니다.
`configs/index_512.yaml`의 sizes는 `[512]`이며, 전용 `index` 동작은 retrieval view 생성이나
QA를 실행하지 않습니다. 질문 513개는 데이터 확인에만 사용합니다.

```bash
conda activate graphrag_chunking
export VLLM_BIN=/home/hottae0/miniconda3/envs/vllm-0101-cu128/bin/vllm
export CUDA_VISIBLE_DEVICES="1"
bash scripts/index_512.sh
```

모델 서버 시작·연결 검사·종료는 기존 `run_server.sh`를 공유합니다. 위 명령은 다음과 같습니다.

```bash
bash scripts/run_server.sh configs/index_512.yaml index
```

모델 서버가 이미 실행 중이면 다음 명령으로 연결 검사 후 인덱싱할 수 있습니다.

```bash
python -m rq1.server index --config configs/index_512.yaml
```

기존 E512 설정·입력·캐시를 검증하고 같은 위치에서 GraphRAG index를 다시 실행합니다.
이는 단계 중간부터 정확히 이어가는 방식이 아니라, 남아 있는 LLM 캐시를 재사용하는 재실행입니다.
완료 그래프는 건너뛰며, 동일 그래프에 대한 동시 인덱싱은 파일 잠금으로 차단합니다.
기존 E128/E256과 비교 조건을 유지하기 위해 서버의 8192토큰 제한과 추출 설정은 변경하지 않았습니다.
따라서 아래에 기록된 컨텍스트 초과 오류가 다시 발생할 가능성은 남아 있습니다.

- 그래프: `runs/_graph_store/novel5_seed42_qwen25_bge_m3/e512`
- 모델 로그: `runs/server_novel5_index_512/model_logs/`
- 인덱싱 상세 로그: 그래프 경로의 `logs/indexing-engine.log`
- 실행 완료 보고서: `runs/server_novel5_index_512/index_complete.json`

`index_complete.json`과 `graph_complete.json`은 파이프라인 완료 표시이며, 개별 추출의 무오류를 보장하지 않습니다.

`run_server.sh`는 실험용 Python 3.11/GraphRAG 3.1.2 환경과 별도 vLLM 실행 파일을
시작 전에 검사합니다. 모델 캐시는 기본적으로 현재 사용자의 `~/.cache/huggingface`를 사용하며,
다른 쓰기 가능한 위치가 필요하면 `MODEL_CACHE_DIR`로 지정합니다. 서버의 전역 CUDA 12.1
라이브러리 경로는 제거하고 vLLM 0.10.1/Torch 2.7.1 cu128 wheel에 포함된 CUDA 12.8
라이브러리를 사용합니다.

실행 중 다른 터미널에서 학습 epoch 대신 GraphRAG stage와 그래프/cell/QA 개수를 확인합니다.

```bash
bash scripts/watch_progress.sh configs/server_preliminary_3x3.yaml
```

화면에는 각 e값의 `WAITING / INDEXING/PARTIAL / COMPLETE`, 최근 GraphRAG workflow 로그,
cache 파일 수, 완료 cell, 처리된 QA 수, 남은 디스크 공간이 10초마다 갱신됩니다.

스크립트는 생성 모델과 임베딩 모델을 터미널에서 선택한 GPU에 올리고, endpoint/structured JSON/임베딩 차원을
검사한 뒤 실험을 실행합니다. 정상 종료, 오류, Ctrl+C 모두에서 스크립트가 자신이 시작한 vLLM
process group을 종료하므로 GPU 메모리가 반환됩니다. 모델 가중치 캐시는 서버 디스크에 남아 다음 실행의
다운로드를 줄입니다. 모델 로그는 결과 폴더의 `model_logs/`에 저장됩니다.

서버 저장소가 `/home/hottae0/Chunking_Granularity`에 있으면 결과는 다음처럼 완전히 분리됩니다.

- 사전실험: `/home/hottae0/Chunking_Granularity/runs/server_novel5_allq_preliminary_3x3` (9 cells)
- 본 실험: `/home/hottae0/Chunking_Granularity/runs/server_novel5_allq_8x8` (64 cells)
- 공용 그래프: `/home/hottae0/Chunking_Granularity/runs/_graph_store/novel5_seed42_qwen25_bge_m3`

각 설정의 `data.graph_store`가 이 공용 저장소를 직접 가리킵니다. 그래프를 실행 결과 폴더에 복사하거나
링크할 필요가 없습니다. 재사용 전에 선택 문서 원문, 모델, 임베딩 차원, chunk size와 overlap을 검사합니다.
완료된 e128/e256은 바로 재사용하고, 부분 완료 e512는 같은 위치에서 재개합니다. 같은 graph store에
대해 두 indexing process를 동시에 실행하면 안 됩니다. 준비 내역은 graph store의
`subset_preparation.json`에 남습니다.

완료 시 `results_complete.json`에 절대 출력 경로, 완료 cell 수와 필수 결과 파일의 절대 경로가
기록됩니다. 중단 후 같은 실행 명령을 다시 실행하면 완료된 그래프와 성공한 질문을 재사용합니다.
동일 output에 실험 process를 동시에 두 개 실행하지 마세요. 실제 속도는 공유 GPU 부하에 영향을 받습니다.

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
5편·선택된 소설의 모든 질문을 쓰는 3×3 사전실험과 8×8 본 실험, 후속 분석은 모두 같은 Qwen2.5-7B 모델로 고정합니다. GPU 연산 사용률이 높은 시간에는
메모리가 남아도 실행 속도가 크게 느려질 수 있습니다.

## 현재 결과와 인수인계 기록 (2026-09-18)

기존 `HANDOFF.md`의 운영 기록은 이 README로 통합했습니다. 기존 미커밋 변경과 실험 결과는 보존합니다.
2×2 전용 실행 스크립트는 제거했으며, `configs/partial_2x2.yaml`은 기존 결과의 설정과 공식 평가용으로 유지합니다.
필요한 경우 `bash scripts/run_server.sh configs/partial_2x2.yaml`로 실행할 수 있습니다.

### 청크 정의와 인덱싱 조건

- Extraction chunk(E)는 그래프 구축 시 엔티티·관계를 추출하는 원문 단위입니다.
- Retrieval chunk(R)는 Local Search가 선택하여 답변 문맥에 넣는 원문 근거 단위입니다.
- 동일 원문을 각 크기로 분할하고, 원문 문자 범위의 겹침으로 그래프 정보와 검색 청크를 연결합니다.
  관계는 대체로 extraction chunk 전체를 출처 범위로 사용하므로 E는 근거 연결 폭에도 영향을 줍니다.
- 같은 E에서 R만 바꿀 때는 동일 그래프를 재사용합니다. 검색 청크별 직접 벡터 검색 실험이 아니라
  엔티티 임베딩과 그래프 연결을 사용하는 GraphRAG Local Search 실험입니다.

실제 중앙 저장소 E128/E256/E512의 `settings.yaml`을 비교한 결과 차이는 `chunking.size`뿐이었습니다.
입력 원문과 프롬프트 파일은 해시 비교에서 동일했습니다. 생성 모델 Qwen2.5-7B-Instruct,
임베딩 BGE-M3(1024차원), tokenizer cl100k_base, overlap 0, seed 42,
인덱싱 동시 요청 수 2와 서버 컨텍스트 제한 8192를 동일하게 사용합니다.

단, 상세 인덱싱 로그에는 아래 오류가 기록되어 있습니다. 숫자는 단계별 오류 로그 건수이며
최종 누락 개수나 고유 실패 청크 수를 확정한 수치는 아닙니다.

| 오류 단계 | E128 | E256 |
| --- | ---: | ---: |
| 그래프 추출 | 1 | 1 |
| Community report 생성 | 53 | 37 |

그래프 추출은 입력 8236/8234토큰이 서버 제한 8192를 넘어 실패했습니다.
추가 추출(gleaning)은 이전 응답을 포함하므로 원문 청크가 작아도 요청이 길어질 수 있습니다.
Community report에는 컨텍스트 초과 및 JSON 형식 검증 실패 등이 있습니다.
설치된 GraphRAG는 그래프 추출 예외 시 빈 엔티티·관계를 반환하고 진행할 수 있습니다.
따라서 조건 설정은 동일하지만 오류의 영향까지 동일하다고 가정하면 안 되며,
파이프라인 완료 및 QA 성공은 인덱싱 전 과정의 성공을 보장하지 않습니다.
현재 Local Search의 community 비중은 0이지만, 실패의 최종 영향은 별도 검증이 필요합니다.

### 완료된 2×2 결과

결과 경로는 `runs/server_novel5_allq_2x2`입니다. 소설 5편의 전체 질문 513개를 사용했으며,
네 셀 각각 고유 질문 513개와 `status=ok` 513개, 총 2052개 결과를 확인했습니다.
완료 시각은 `2026-09-18T02:56:36Z`, `results_complete.json`은 4/4입니다.
검색 문맥 파싱은 전체 성공했습니다. 실행 중 진행률 회귀 후 최종 행 수·고유성을 다시 확인했으며,
동일 output의 중복 실행 방지 잠금이 추가되어 있습니다.

| E / R | Answer F1 | Evidence@5 proxy | Relation recall proxy | Path coverage proxy | 평균 지연(초) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 / 128 | 0.275343 | 0.802171 | 0.918837 | 0.847953 | 3.457 |
| 128 / 256 | 0.281282 | 0.890806 | 0.918837 | 0.847953 | 3.334 |
| 256 / 128 | 0.274186 | 0.756074 | 0.893502 | 0.810916 | 2.668 |
| 256 / 256 | 0.280667 | 0.874724 | 0.893502 | 0.810916 | 1.769 |

R128→R256의 F1 변화는 E128에서 +0.005939(CI [0.003104, 0.008845]),
E256에서 +0.006480(CI [0.000576, 0.010908])입니다.
최고 셀 E128/R256과 E256/R256의 차이는 0.000615, paired bootstrap CI는
[-0.007606, 0.008604]로 E의 우열은 불명확합니다. 질문별 승/패/동률은 250/251/12입니다.
Bootstrap winner 비중은 각각 0.615/0.385이며 네 셀 모두 best 대비 0.02 이내입니다.
소설별로 E128/R256이 3/5, E256/R256이 2/5에서 우세했습니다.
질문 유형별 최고는 Fact retrieval·Complex reasoning에서 E128/R256,
Context summarization에서 E256/R256, Creative generation에서 E128/R128(n=24)입니다.
지연은 순차 실행의 warm-up/cache/order 및 공유 GPU 영향을 포함하므로 인과적 속도 차이로 해석하지 않습니다.

공식 judge 점수는 아직 없으며, EM/accuracy proxy는 전부 0입니다.
정확 evidence span 지표도 원문에서 evidence 문장을 찾지 못해 미채점입니다.
현재 F1과 lexical proxy는 탐색적 비교용이며, 위 인덱싱 오류와 작은 크기 범위를 함께 고려해야 합니다.
공식 평가 전에는 `official_eval.py`의 현재 output 연결과 task별 지원 지표를 검토하세요.

```bash
python -m rq1.experiments.official_eval --config configs/partial_2x2.yaml \
  --benchmark-root /path/to/GraphRAG-Benchmark \
  --judge-model YOUR_JUDGE --embedding-model /path/to/bge-model
```

### 저장소 및 환경 상태

중앙 저장소는 `runs/_graph_store/novel5_seed42_qwen25_bge_m3`입니다.
E128(약 96MB)과 E256(약 66MB)은 완료, E512(약 29MB)는 community report 생성 중
디스크 부족으로 중단되어 캐시가 남아 있습니다. E512의 `community_reports.parquet`과
`graph_complete.json`은 아직 없습니다. E768/1024/1200/1536/2048은 placeholder입니다.
과거 결과의 일부 `graphs/e*` 심볼릭 링크는 이 저장소를 가리킵니다.
과거 중복 캐시·오래된 결과 정리로 약 1.1GB를 확보했으며 추가 삭제 전에는 현재 필요 여부를 확인하세요.
2026-09-18 재확인 시 파일시스템 여유 공간은 약 428GB였습니다(실행 전 `df -h .`로 재확인).

실험 환경은 `/home/hottae0/miniconda3/envs/graphrag_chunking`의 Python 3.11.16,
GraphRAG 3.1.2, NumPy 2.4.6입니다. 모델 서버는 별도 환경
`/home/hottae0/miniconda3/envs/vllm-0101-cu128`의 vLLM 0.10.1,
PyTorch 2.7.1+cu128, Transformers 4.55.0을 사용합니다.
드라이버 570.181은 CUDA 12.8을 지원하며 시스템 CUDA toolkit 12.1과의 라이브러리 충돌을
피하기 위해 실행 스크립트에서 `LD_LIBRARY_PATH`를 해제합니다.
Hugging Face 캐시는 `/home/hottae0/.cache/huggingface`이며 모델 가중치는 합계 약 17GB입니다.

다음 연구 단계는 E512 완성, 동일 소설 5편·전체 질문 513개를 유지한 확장 실험,
공식 judge 평가와 최종 표·그래프 갱신입니다. 현재 변경 내역은 `git status --short`로 확인하세요.
