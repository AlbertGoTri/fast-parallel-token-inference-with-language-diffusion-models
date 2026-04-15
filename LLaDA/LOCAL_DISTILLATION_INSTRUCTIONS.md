# Local Distillation & Profiling Instructions

This repository includes a pipeline to perform knowledge distillation from a teacher model to a LoRA-adapted student model. The pipeline focuses on predicting diffusion steps and text generation. You can find the following essential scripts:

* **[`01_cache_teacher.py`](01_cache_teacher.py)**: Executes inference using the teacher model and builds cached trajectories (intermediate generation values).
* **[`02_train_student.py`](02_train_student.py)**: Uses the cached trajectories to train a student via knowledge distillation with LoRA adapters. Automatically optimizes via KL divergence against the collected teacher logits.
* **[`run_llada_local.py`](run_llada_local.py)**: A local evaluator script to run generation inference on either the foundation model or the fine-tuned LoRA student.

## Complete Pipeline Execution

1. **Generate Teacher Trajectories**: 
   Run `01_cache_teacher.py` to prompt the teacher model and intercept intermediate states during the generation of text. Caches these values as `.pt` bundles.
2. **Train the Student Model**: 
   Run `02_train_student.py` to train the LoRA student using the collected trajectory caches.
3. **Evaluate Performance locally**: 
   Run `run_llada_local.py` to interact with and test the generation capabilities of your fine-tuned student model natively.

## Instrumenting and Profiling Execution

To assist in thesis writing, hardware usage analysis, and execution performance validation, all three scripts mentioned above are instrumented with the built-in PyTorch Profiler (`torch.profiler.profile`).

By executing these scripts, tracing and execution-time summaries are automatically produced and published to the `profiling/` directory:

* **Performance Summaries** (e.g., `profiling/01_cache_teacher_summary.txt`):
  Human-readable aggregated tables representing CPU vs CUDA execution time overhead, sorting operations mathematically by their GPU load lengths. 
* **Chrome Traces** (e.g., `profiling/01_cache_teacher_trace.json`):
  Next-generation timeline trace bundles. You can natively load these into `chrome://tracing` or Microsoft Edge (`edge://tracing`) to review the execution timeline visually alongside CPU & GPU interactions.