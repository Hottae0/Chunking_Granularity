"""Direct execution on an allocated GPU host; no scheduler or GPU reassignment."""
import argparse
import json
import math
import os
import time
from pathlib import Path
from rq1.config import load_settings


def serve_command(kind, settings, tensor_parallel=1, memory=None, max_model_len=8192, port=None):
    embedding = kind == 'embedding'
    # Lab server default: chat and embedding share one allocated H100.
    memory = (0.10 if embedding else 0.55) if memory is None else memory
    if tensor_parallel < 1 or not 0 < memory < 1 or max_model_len < 1:
        raise ValueError('Invalid GPU serving parameters')
    model = settings.embedding_model if embedding else settings.model
    if model in ('unset', 'your-chat-model', 'your-embedding-model'):
        raise ValueError('Set MODEL and EMBEDDING_MODEL in .env first')
    command = [os.getenv('VLLM_BIN', 'vllm'), 'serve', model,
               '--host', '127.0.0.1', '--port', str(port or (8001 if embedding else 8000)),
               '--tensor-parallel-size', str(tensor_parallel),
               '--gpu-memory-utilization', str(memory), '--max-model-len', str(max_model_len)]
    if embedding:
        command += ['--runner', 'pooling']
    else:
        command += ['--generation-config', 'vllm']
    return command


def wait_models(client, model, timeout):
    from openai import APIConnectionError, APITimeoutError, APIStatusError
    deadline = time.monotonic() + timeout
    while True:
        try:
            available = {x.id for x in client.models.list().data}
            if model not in available:
                raise ValueError(f'Model {model!r} is not served; available models: {sorted(available)}')
            return
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            if isinstance(exc, APIStatusError) and exc.status_code < 500: raise
            if time.monotonic() >= deadline:
                raise RuntimeError('Model server readiness timed out') from exc
            time.sleep(min(2, max(0, deadline-time.monotonic())))


def preflight(settings, timeout=600, factory=None):
    if settings.backend != 'official': raise ValueError('Server runner requires backend: official')
    if not settings.corpus.exists() or not settings.questions.exists():
        raise FileNotFoundError('Prepare corpus and questions before starting the GPU experiment')
    if settings.embedding_dimensions <= 0 or settings.indexing_concurrency <= 0:
        raise ValueError('Embedding dimensions and indexing concurrency must be positive')
    if factory is None:
        from openai import OpenAI
        factory = OpenAI
    chat = factory(base_url=settings.base_url, api_key=settings.api_key, timeout=30, max_retries=0)
    embedding = factory(base_url=settings.embedding_base_url, api_key=settings.embedding_api_key,
                        timeout=30, max_retries=0)
    try:
        wait_models(chat, settings.model, timeout)
        wait_models(embedding, settings.embedding_model, timeout)
        result = embedding.embeddings.create(model=settings.embedding_model, input=['Connection test.'])
        vector = result.data[0].embedding
        if len(vector) != settings.embedding_dimensions or not all(math.isfinite(x) for x in vector):
            raise ValueError(f'Embedding dimensions mismatch: server={len(vector)}, config={settings.embedding_dimensions}; set EMBEDDING_DIMENSIONS')
        result = chat.chat.completions.create(model=settings.model, temperature=0, max_tokens=64,
            messages=[dict(role='user', content='Return JSON with exactly one field: ok, set to true.')],
            response_format={'type':'json_schema','json_schema':{'name':'readiness','strict':True,
                'schema':{'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False}}})
        if json.loads(result.choices[0].message.content).get('ok') is not True:
            raise ValueError('Chat server failed structured JSON output probe')
        return dict(chat_model=settings.model, embedding_model=settings.embedding_model,
                    embedding_dimensions=len(vector), structured_output=True,
                    cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
                    note='Readiness probes only; complete a pilot before the full run')
    finally:
        chat.close()
        embedding.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['serve-chat','serve-embedding','check','run'])
    parser.add_argument('--config', type=Path, default=Path('configs/server.yaml'))
    parser.add_argument('--wait-seconds', type=float, default=600)
    parser.add_argument('--tensor-parallel', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=None,
                        help='Per-process vLLM fraction; defaults to 0.55 chat / 0.10 embedding')
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--port', type=int)
    args=parser.parse_args()
    settings=load_settings(args.config)
    if args.action.startswith('serve-'):
        command=serve_command(args.action.removeprefix('serve-'), settings,
                              args.tensor_parallel, args.gpu_memory_utilization, args.max_model_len, args.port)
        print('Starting model server with inherited CUDA_VISIBLE_DEVICES.', flush=True)
        # exec keeps terminal signals attached to the model server; no orphan background processes.
        os.environ['VLLM_API_KEY'] = settings.embedding_api_key if args.action == 'serve-embedding' else settings.api_key
        os.execvp(command[0], command)
    if args.wait_seconds <= 0: parser.error('--wait-seconds must be positive')
    report=preflight(settings, args.wait_seconds)
    print(json.dumps(report, indent=2))
    if args.action == 'run':
        from rq1.experiments.run_grid import run
        settings.output.mkdir(parents=True, exist_ok=True)
        report['started_at_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        report_dir=settings.output/'server_checks'
        report_dir.mkdir(exist_ok=True)
        (report_dir/f'{time.time_ns()}.json').write_text(json.dumps(report, indent=2))
        output=run(args.config)
        manifest=json.loads((output/'run_manifest.json').read_text())
        if manifest['completed_cells'] != manifest['expected_cells']:
            raise SystemExit('Incomplete experiment: inspect failed questions in per_query_results.csv and rerun')
        print(output)

if __name__ == '__main__': main()
