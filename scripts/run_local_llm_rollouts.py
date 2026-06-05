#!/usr/bin/env python
"""Run tau2 text rollouts with local OpenAI-compatible LLM endpoints.

Example:
    python scripts/run_local_llm_rollouts.py \
        --domain mock \
        --agent-model Qwen/Qwen2.5-7B-Instruct \
        --agent-api-base http://127.0.0.1:8000/v1 \
        --user-model Qwen/Qwen2.5-7B-Instruct \
        --user-api-base http://127.0.0.1:8001/v1 \
        --num-tasks 2 \
        --output-dir data/local_rollouts/demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tau2.data_model.simulation import Results, TextRunConfig
from tau2.run import run_domain


def _json_arg(value: Optional[str]) -> dict[str, Any]:
    if not value:
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object")
    return loaded


def _local_model_name(model: str) -> str:
    if model.startswith("openai/"):
        return model
    return f"openai/{model}"


def _llm_args(
    *,
    api_base: Optional[str],
    api_key: Optional[str],
    temperature: float,
    max_tokens: Optional[int],
    extra: dict[str, Any],
) -> dict[str, Any]:
    args: dict[str, Any] = {"temperature": temperature}
    if max_tokens is not None:
        args["max_tokens"] = max_tokens
    if api_base:
        args["api_base"] = api_base
    if api_key:
        args["api_key"] = api_key
    args.update(extra)
    return args


def _message_to_record(message) -> dict[str, Any]:
    record = message.model_dump(mode="json")
    if record.get("raw_data") is not None:
        record["raw_data"] = None
    return record


def _write_trajectories(results: Results, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as fp:
        for sim in results.simulations:
            messages = [_message_to_record(message) for message in sim.get_messages()]
            record = {
                "task_id": sim.task_id,
                "trial": sim.trial,
                "reward": sim.reward_info.reward if sim.reward_info else None,
                "termination_reason": sim.termination_reason.value,
                "messages": messages,
            }
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tau2 rollouts with separate local agent and user LLMs."
    )
    parser.add_argument("--domain", default="mock")
    parser.add_argument("--task-set-name", default=None)
    parser.add_argument("--task-split-name", default="base")
    parser.add_argument("--task-ids", nargs="+", default=None)
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-errors", type=int, default=5)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path("data/local_rollouts"))

    parser.add_argument(
        "--agent-model",
        default=os.getenv("TAU2_LOCAL_AGENT_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    )
    parser.add_argument(
        "--agent-api-base",
        default=os.getenv("TAU2_LOCAL_AGENT_API_BASE", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument(
        "--agent-api-key",
        default=os.getenv("TAU2_LOCAL_AGENT_API_KEY", "EMPTY"),
    )
    parser.add_argument("--agent-temperature", type=float, default=0.2)
    parser.add_argument("--agent-max-tokens", type=int, default=None)
    parser.add_argument("--agent-llm-args", type=_json_arg, default={})

    parser.add_argument(
        "--user-model",
        default=os.getenv("TAU2_LOCAL_USER_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    )
    parser.add_argument(
        "--user-api-base",
        default=os.getenv("TAU2_LOCAL_USER_API_BASE", "http://127.0.0.1:8001/v1"),
    )
    parser.add_argument(
        "--user-api-key",
        default=os.getenv("TAU2_LOCAL_USER_API_KEY", "EMPTY"),
    )
    parser.add_argument("--user-temperature", type=float, default=0.7)
    parser.add_argument("--user-max-tokens", type=int, default=None)
    parser.add_argument("--user-llm-args", type=_json_arg, default={})
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TextRunConfig(
        domain=args.domain,
        task_set_name=args.task_set_name,
        task_split_name=args.task_split_name,
        task_ids=args.task_ids,
        num_tasks=args.num_tasks,
        num_trials=args.num_trials,
        agent="llm_agent",
        user="local_user_simulator",
        llm_agent=_local_model_name(args.agent_model),
        llm_args_agent=_llm_args(
            api_base=args.agent_api_base,
            api_key=args.agent_api_key,
            temperature=args.agent_temperature,
            max_tokens=args.agent_max_tokens,
            extra=args.agent_llm_args,
        ),
        llm_user=_local_model_name(args.user_model),
        llm_args_user=_llm_args(
            api_base=args.user_api_base,
            api_key=args.user_api_key,
            temperature=args.user_temperature,
            max_tokens=args.user_max_tokens,
            extra=args.user_llm_args,
        ),
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        save_to=None,
        log_level="INFO",
    )

    results = run_domain(config)
    results_path = output_dir / "results.json"
    trajectories_path = output_dir / "trajectories.jsonl"
    results_path.write_text(results.model_dump_json(indent=2), encoding="utf-8")
    _write_trajectories(results, trajectories_path)

    summary = {
        "num_simulations": len(results.simulations),
        "results_path": str(results_path),
        "trajectories_path": str(trajectories_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
