# AGENTS.md

Guidance for AI agents and contributors working in this repository.

## Project overview

`transtab` is a flexible tabular prediction framework based on the TransTab paper
(Zifeng Wang et al.). It represents each table row as a sequence of tokens
(column name + value) and processes it with a transformer encoder.

This fork adds an **encoder–decoder (autoencoder) extension**:

- `transtab.build_encoder` builds a standalone transformer encoder whose
  `forward(x)` returns the `[CLS]` summary embedding `(bs, hidden_dim)`.
- `transtab.build_decoder` builds a decoder that reconstructs the **original
  table format** from the `[CLS]` embedding: `decoder(encoder(x))` returns a
  DataFrame with the same columns as `x`.
- The decoder is trained jointly with the encoder **only** when the model is
  created via `transtab.build_supcon_bce_learner` (SupCon + BCE + reconstruction
  loss) and trained with `transtab.train`.

## Environment (required)

**Always use the conda environment `project`** to run scripts, tests, or
one-off commands. The base environment and other envs (`german`, `aoc25`,
`dashboard`) do not have the required dependencies.

```bash
/Users/noel/miniconda3/envs/project/bin/python        # Python 3.10.9
conda activate project                                 # equivalent interactive use
```

Key versions in `project`: torch 2.8.0, transformers 4.30.0, pandas 2.2.2.

Example:

```bash
cd /Users/noel/Documents/RPTU/Winter2025/Thesis/Tabular/repo/transtab
/Users/noel/miniconda3/envs/project/bin/python examples/encoder_decoder_demo.py --epochs 1
```

## Importing the local package (gotcha)

The env also contains a **pip-installed `transtab` 0.0.7** in site-packages.
When running `python some_dir/script.py`, Python puts the script's directory
(not the repo root) on `sys.path`, so `import transtab` silently resolves to the
installed copy — which does NOT contain this repo's changes.

Rules of thumb:

- Verify which package you're importing with `python -c "import transtab; print(transtab.__file__)"` — it must print a path under this repo, not site-packages.
- When running scripts from subdirectories (e.g. `examples/`), insert the repo root into `sys.path` before importing, exactly like `examples/encoder_decoder_demo.py` does.
- Alternatively run one-off commands from the repo root with `python -c ...` (the repo root is on `sys.path` when CWD is the repo root).

## Repository layout

```
transtab/
  transtab.py            # public API: build_* helpers, train()
  modeling_transtab.py   # all model classes (encoder, decoder, heads, losses)
  trainer.py             # Trainer (training loop, optimizer, early stopping)
  trainer_utils.py       # collators, datasets, schedulers
  dataset.py             # load_data / load_single_data (local or OpenML)
  evaluator.py           # predict, evaluate, metrics, EarlyStopping
  constants.py           # checkpoint file names (WEIGHTS_NAME, DECODER_NAME, ...)
examples/
  encoder_decoder_demo.py   # encoder->decoder round-trip demo on Adult data
  *.ipynb                   # older notebooks (classifier / CL / transfer learning)
data/                        # gitignored; contains Original/ and Processed/ datasets
build/                       # stale build artifact — do NOT edit
```

## Architecture notes (relevant to the decoder work)

- **Tokenization**: `TransTabFeatureExtractor` turns a DataFrame into
  `x_num`/`num_col_input_ids`/`x_cat_input_ids`/`x_bin_input_ids` using a BERT
  tokenizer. `TransTabFeatureProcessor` maps those to token embeddings.
- **Encoder**: `TransTabInputEncoder` (extractor + processor) → `TransTabCLSToken`
  (prepends a learnable `[CLS]` at position 0) → `TransTabEncoder` (transformer
  stack). `TransTabModel.forward(x)` returns `encoder_output[:, 0, :]` — the CLS.
- **Decoder** (`TransTabDecoder`): takes the CLS embedding, expands it to one
  slot per column via learned `col_queries`, refines with transformer layers,
  and predicts each column value with per-column heads (regression for numerical,
  classification over a training-collected vocabulary for categorical, logistic
  for binary). `forward(h)` returns a pandas DataFrame; `loss(h, x)` is the
  differentiable reconstruction loss.
- **Categorical vocabulary**: the decoder's category vocabularies are collected
  from the training tables by `TransTabForSupConBCE.prepare_decoder`, which the
  `Trainer` calls (if the method exists) **before creating the optimizer**, so
  the lazily-built head parameters are trained. Vocab + schema + weights are
  saved in `decoder.bin` and restored by `build_decoder(checkpoint=...)`.
- **Training flow**: `transtab.train` → `Trainer` → for each batch
  `logits, loss = model(data[0], data[1])` → `loss.backward()`. The decoder is
  trained only because (a) it lives inside `TransTabForSupConBCE` (built only by
  `build_supcon_bce_learner`) and (b) its reconstruction loss is part of the
  model's total loss. `TransTabCollatorForSupConBCE` passes the raw table along
  as `x_raw` so the recon loss can be computed against original values.

## Running the example

```bash
cd /Users/noel/Documents/RPTU/Winter2025/Thesis/Tabular/repo/transtab
/Users/noel/miniconda3/envs/project/bin/python examples/encoder_decoder_demo.py --epochs 1   # quick smoke test
/Users/noel/miniconda3/envs/project/bin/python examples/encoder_decoder_demo.py              # 5 epochs
```

The demo trains on `data/Original/Adult_Data` (~48k rows), then round-trips
held-out rows through `build_encoder`/`build_decoder` and prints numerical MAE /
categorical accuracy / binary accuracy. Checkpoints land in `./ckpt_decoder_demo`
by default (override with `--output-dir`).

## Conventions and gotchas

- `data/` is gitignored; the Adult dataset lives at `data/Original/Adult_Data`
  (only present on this machine).
- `build/` is a stale copy of the package from an earlier `pip install` — never
  edit it; edit `transtab/` sources.
- The model's `load`/`save` use non-strict `load_state_dict` (`strict=False`);
  loading a checkpoint with mismatched `hidden_dim`/`num_layer` silently drops
  keys — keep architecture kwargs identical across `build_supcon_bce_learner`,
  `build_encoder`, and `build_decoder`.
- `transtab/tokenizer/` is a local cache of the BERT tokenizer created on first
  use — safe to delete; it will be re-downloaded/re-saved.
- When adding new scripts, follow the `examples/encoder_decoder_demo.py` pattern:
  repo-root `sys.path` insertion, `conda env project` interpreter, data path
  relative to the script.
