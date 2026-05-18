## Phase 1: Foundation ✅

- [x] Project scaffolding & directory structure
- [x] Config-driven IMDB training pipeline
- [x] Core interfaces & protocols
- [x] Registry pattern, data loaders, model factory
- [x] Training, inference, evaluation modules
- [x] End-to-end pipeline abstractions

## Phase 2: Experiments

- [ ] **Baselines** — TF-IDF + Logistic Regression lower-bound
- [ ] **HF Transformers** — Config-driven fine-tuning with model overrides
- [ ] **LoRA** — PEFT-based parameter-efficient fine-tuning
- [ ] **RAG** — FAISS index build + retrieval query
- [ ] **Agents** — LLM tool-calling agent loop

## Phase 3: Application

- [ ] **Gradio** — Interactive text classification demo
- [ ] **FastAPI** — Inference REST API with `/predict` endpoint

## Phase 4: Production

- [ ] Model versioning & experiment tracking (MLflow / W&B)
- [ ] Distributed training (multi-GPU / FSDP)
- [ ] ONNX / TensorRT model export
- [ ] CI/CD pipeline for training & evaluation
- [ ] Docker deployment
