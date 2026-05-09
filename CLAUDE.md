# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

AI 模型工程实验平台 — a structured workspace for ML model training, inference, evaluation, and app building. The package name under `src/` is `ai_model_engineering_lab`.

## Conventions

- All file and directory names use `snake_case`
- Configs are YAML, stored under `configs/` grouped by purpose (base, training, inference, evaluation)
- Experiment code lives in `experiments/` organized by technical topic (lora, rag, agents, etc.)
- Data pipeline: `data/raw/` → `data/interim/` → `data/processed/`, with `data/external/` for third-party data
- Outputs are written to `outputs/` (checkpoints, logs, metrics, predictions) — never to the source tree
