# Chunking Granularity — GraphRAG RQ1

Graph extraction chunk 크기 `g_E`와 evidence retrieval chunk 크기 `g_R`를 독립적으로 바꾸어
GraphRAG QA 성능을 측정하는 실험 저장소입니다. 현재 구현은 Microsoft GraphRAG 3.1.2를
지원하며, 데이터셋과 백엔드를 추가할 수 있도록 실행 코드에서 분리했습니다.

## 핵심 정의

- Extraction chunk(E): 엔티티와 관계를 추출하여 그래프를 만드는 원문 단위입니다.
- Retrieval chunk(R): 질문 시 Local Search가 선택해 답변 문맥에 넣는 원문 단위입니다.
- 같은 E의 인덱스는 모든 R 조건에서 한 번만 저장하고 재사용합니다.
- E에서 추출된 fact는 원문 문자 offset의 겹침으로 R chunk에 연결합니다.

관계 provenance는 현재 관계가 추출된 E chunk 전체 범위를 사용합니다. 따라서 E 효과에는
순수 extraction quality뿐 아니라 E→R mapping 범위의 변화도 포함됩니다. extraction 효과만
분리하려면 관계를 지지하는 문장 수준 offset을 E와 독립적으로 식별해야 합니다.

## 저장소 구조

```text
configs/
  experiments/   # grid, 데이터셋/백엔드 선택, output/index 경로
  datasets/      # corpus/questions 위치
  backends/      # 검색·인덱싱 기본값
data/
  raw/           # 원본 데이터; Git 제외
  processed/     # 선택적 전처리 결과; Git 제외
  synthetic/     # 선택적 합성 데이터
src/rq1/
  core/          # 공통 타입, token chunking, provenance, alignment
  datasets/      # 데이터셋 loader와 registry
  backends/      # backend registry, MS GraphRAG, HippoRAG scaffold
  eval/          # QA/retrieval/relation 지표
  experiments/   # grid, index-only, QA-only, 분석과 공식 평가
  llm/           # LLM client
  plotting/      # figure 생성
scripts/
  download/      # 데이터 다운로드
  run/           # 실행 진입점
  utilities/     # 진행률 및 index import 도구
runs/
  indexes/<backend>/<dataset>/e<size>/
  experiments/<experiment>/
```

`data/`와 `runs/`는 `.gitignore`로 제외합니다.

## 2026-09-18 구조 개편

이번 개편에서 다음을 변경했습니다.

- `chunking`, `alignment`, `data`, `msgraphrag`, `retrieval`에 흩어져 있던 코드를
  `core`, `datasets`, `backends`로 재배치했습니다.
- experiment/dataset/backend 설정을 각각의 YAML로 분리했습니다.
- 모든 MS GraphRAG index를 `runs/indexes/ms_graphrag/<dataset>/e<size>`에 한 번만 저장합니다.
- 실험 산출물은 `runs/experiments/<experiment>`에 분리하고 figure는 `figures/`에 둡니다.
- QA-only 실행은 완료된 index만 허용하며 인덱싱을 암묵적으로 시작하지 않습니다.
- 이전 grid의 완료 cell은 행 수·질문 ID·E/R·상태·prediction을 검증한 후 심볼릭 링크로
  연결합니다. 복사와 중복 QA 호출을 피합니다.
- 3×3 설정은 기존 2×2의 네 cell을 필수 재사용 대상으로 지정합니다. 파일이 없으면
  조용히 2,052개 QA를 다시 수행하지 않고 실행 전에 중단합니다.
- index별 파일 잠금과 experiment별 실행 잠금을 유지해 중복 writer를 차단합니다.
- Local Search의 context/community 비중은 재사용한 index의 과거 설정이 아니라 현재 backend
  설정으로 덮어써 E 조건마다 동일하게 적용합니다.
- HippoRAG와 MuSiQue/2Wiki/QASPER는 registry와 명확한 미구현 오류만 준비했습니다.
  아직 실행 가능한 구현으로 표시하지 않습니다.

## 설치와 환경

실험 환경은 Python 3.11과 GraphRAG 3.1.2를 사용합니다.

```bash
conda activate graphrag_chunking
python -m pip install -e .
cp .env.example .env
```

`.env` 예시:

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

데이터가 없다면 다음을 실행합니다.

```bash
bash scripts/download/download_graphrag_bench.sh
```

파일은 `data/raw/graphrag_bench/Datasets/...`에 저장됩니다.

## 설정 구성

실험 설정은 데이터셋과 백엔드 이름을 참조합니다.

```yaml
experiment:
  name: rq1_graphrag_bench_msgraphrag_3x3
  dataset: graphrag_bench
  backend: ms_graphrag
  sizes: [128, 256, 512]
  output: runs/experiments/rq1_graphrag_bench_msgraphrag_3x3
  index_store: runs/indexes/ms_graphrag/graphrag_bench
```

현재 실험 설정:

| 설정 | 크기 | 용도 |
| --- | --- | --- |
| `rq1_pilot.yaml` | 256, 512, 1200, 2048 | 작은 연결 점검 |
| `rq1_2x2.yaml` | 128, 256 | 기존 2×2 |
| `rq1_3x3.yaml` | 128, 256, 512 | E512 확장, 기존 4 cell 재사용 |
| `rq1_full.yaml` | 8개 크기 | 전체 8×8 |

## 실행

모델 서버까지 자동 관리하는 실행:

```bash
export VLLM_BIN=/home/hottae0/miniconda3/envs/vllm-0101-cu128/bin/vllm
export CUDA_VISIBLE_DEVICES=1

# pilot: index + QA
bash scripts/run/run_pilot.sh

# 전체 실험: index + QA
bash scripts/run/run_full.sh
```

동일 runner에서 동작을 선택할 수 있습니다.

```bash
# E128/E256/E512 index 확인·생성만 수행
bash scripts/run/run_server.sh configs/experiments/rq1_3x3.yaml index

# 완료된 index를 사용해 3×3 QA만 수행
bash scripts/run/run_server.sh configs/experiments/rq1_3x3.yaml qa

# 필요한 index를 만들고 QA까지 수행
bash scripts/run/run_server.sh configs/experiments/rq1_3x3.yaml run
```

이미 모델 endpoint가 실행 중이면 Python 진입점을 직접 사용할 수 있습니다.

```bash
python -m rq1.experiments.index_only --config configs/experiments/rq1_3x3.yaml
python -m rq1.experiments.qa_only --config configs/experiments/rq1_3x3.yaml
python -m rq1.experiments.run_grid --config configs/experiments/rq1_full.yaml
```

진행률:

```bash
python scripts/utilities/progress.py --config configs/experiments/rq1_3x3.yaml
```

## 캐시와 저장 공간

현재 GraphRAG-Bench/MS GraphRAG index 경로는 다음 하나입니다.

```text
runs/indexes/ms_graphrag/graphrag_bench/
  e128/
  e256/
  e512/
  ...
```

index 재사용 전 다음을 검사합니다.

- extraction size와 overlap
- chat/embedding model
- embedding 차원
- 선택 문서 이름과 원문 내용
- 필수 parquet table과 `graph_complete.json`

실험 cell 캐시는 `cache_identity.json`으로 데이터, 질문, 과학적 설정, 패키지 버전과 코드
fingerprint를 검사합니다. index 위치와 이전 cell 위치는 저장 위치일 뿐이므로 scientific
fingerprint에서 제외합니다. 기존 cell은 복사하지 않고 symlink로 연결하며 `cell_reuse.json`에
출처를 기록합니다.

현재 로컬 index 상태는 E128/E256/E512 모두 complete입니다. 구조 이전 시 기존 2×2 폴더에는
완료 표식과 요약은 남아 있었지만 `cells/`, `per_query_results.csv`, `config_summary.csv`가 이미
없었습니다. 따라서 현재 디스크에서 재사용 가능한 2×2 cell은 0개입니다. 잘못된 완료 판단을
막기 위해 과거 완료 파일은 `results_complete.stale.json`으로 보존했고, 3×3은 네 cell 복구 또는
명시적인 재실행 선택 전까지 preflight에서 중단합니다.

## 결과 형식

실험 결과는 다음과 같습니다.

```text
runs/experiments/<name>/
  manifest.json
  cache_identity.json
  cell_reuse.json
  cells/e<E>_r<R>/
    per_query.csv
    benchmark_predictions.json
    provenance.parquet
    output/
  per_query_results.csv
  config_summary.csv
  question_type_summary.csv
  bootstrap_ci.json
  rq1_analysis.json
  near_optimal_cells.csv
  figures/
```

Answer F1과 lexical evidence/relation 지표는 탐색적 proxy입니다. 최종 결론은 공식 judge 평가를
추가한 뒤 내려야 합니다.

```bash
python -m rq1.experiments.official_eval \
  --config configs/experiments/rq1_3x3.yaml \
  --benchmark-root /path/to/GraphRAG-Benchmark \
  --judge-model YOUR_JUDGE \
  --embedding-model /path/to/bge-model
```

## 검증

```bash
python -m unittest discover -s tests -v
bash -n scripts/download/*.sh scripts/run/*.sh
```

실행 환경과 검증 기록은 [VALIDATION.md](VALIDATION.md)를 확인하세요.
