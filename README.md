# When CAPTCHA Bots Behave Like Humans: Limits of Server-Side Detection

Code, results and literature corpus for the paper

> N. Kekatos, T. Nestoridis, G. Koutidis, M. Curca, A. Lekidis and T. Nianios,
> "When CAPTCHA Bots Behave Like Humans: Limits of Server-Side Detection,"
> 3rd Int. Workshop on Big Data Analysis and Illicit Trends (BANDIT 2026), IEEE BigData 2026, Phoenix, AZ, USA. To appear.

## Question

A site that serves its own CAPTCHAs can record, for every challenge, when it was answered, how long the
client took and whether the answer was correct. How far can the site tell bots from people with only these
records, when the attacker also makes its bot's timing look human?

## Main findings

| Research question | Finding |
|---|---|
| RQ1: what do the usual defences cost a low-budget solver? | A small CRNN trained on the deployed distortion level reads 92.4% of the hardest level. After a change of scheme, 100 bought labels raise a transferred solver from 0.4% to 57.7%. Adversarial CAPTCHAs fool the defender's own model (0.7%) but not an independent solver (83.8% at ε = 8/255). A local VLM (Qwen3.5 9B) reads 98.7% of plain and 46.0% of unseen-scheme CAPTCHAs without training. |
| RQ2: does a monitor over timing, correctness and rate detect these solvers within a 1% false-positive budget? | Yes for bots that do not slow down: every one is flagged within 30 challenges per key, while 0.7% of held-out people are flagged (monitor without the dispersion check). Recalibrated to a realised 0.8% false-positive rate, the full monitor also flags 42.7% of bots that replay solve times pooled from many people. |
| RQ3: what remains when the bot imitates human timing? | A bot that samples the solving times of one person, with human accuracy, is flagged no more often than a person (0.8%), and no detector over the same records can do better (Proposition 1). Only the hourly cap per key remains: 25 challenges, i.e. at most 24.8 correct solves per key and hour, so one million solves per day need 1,682 keys instead of 23. |

All human false-positive rates are estimates from a model fitted to a public human study (Searles et al.,
2023) with three solves per person.

## Layout

| Path | Content |
|---|---|
| `code/gen.py` | text-CAPTCHA generators (distortion levels L0–L3, schemes G1, G2) |
| `code/solver.py`, `code/exp.py` | CRNN-CTC solver, PGD perturbations; experiments E1, E1b, E2, E3 |
| `code/vlm.py` | E4, zero-shot VLMs through a local Ollama server |
| `code/monitor.py`, `code/run_monitor.py`, `code/matched_fpr.py`, `code/review1027.py` | E5, the runtime monitor (properties P1–P6), human model, calibration, attacker profiles, realised-1% operating point, telemetry faults, total-variation estimates |
| `code/macros.py`, `code/macros_review1027.py`, `code/corpus.py`, `code/plots.py` | turn results into the LaTeX macros and figures of the paper |
| `results/*.json`, `results/logs/` | raw results of every experiment and their run logs |
| `results/RESULTS.md` | every number in the paper with its source |
| `paper/*.tex` | the generated macro files the paper is built from |
| `lit/` | coded literature corpus (144 works, slices A–C), BibTeX, and the 21 Oct 2026 solving-service price snapshot |
| `data_ext/` | source commit of the human data (see below) |

## Human data

The human solve times come from the public raw data of Searles et al., "An Empirical Study and Evaluation of
Modern CAPTCHAs" (USENIX Security 2023), https://github.com/sprout-uci/captcha-study, commit in
`data_ext/SOURCE_COMMIT.txt`. That repository carries no licence, so the extracted file
`data_ext/searles2023_text_times.csv` (columns `pid,bias,simple_s,moving_s,masked_s`: the three text-CAPTCHA
solve times per participant, in seconds) is not redistributed here. Extract it from the raw data or ask the
authors.

Trained model weights are not included.

## Reproduce

Python 3.14, packages in `requirements.txt` (PyTorch with MPS or CUDA). E4 needs `ollama serve` with
`qwen3.5:9b` and `gemma3:4b`.

```
cd code
python exp.py e1; python exp.py e1b; python exp.py e3; python exp.py e2
python vlm.py qwen3.5:9b 150; python vlm.py gemma3:4b 150
python run_monitor.py
python review1027.py
python corpus.py; python plots.py; python macros.py; python macros_review1027.py
```

## AI assistance

The experiment code, the literature searches and the coding of the corpus were produced with a generative AI
assistant (Anthropic's Claude) under the authors' direction, as stated in the paper's acknowledgment.

## Funding

This work has received funding from the European Union's Horizon Europe research and innovation programme
under grant agreement No 101309530 (CyberINTEL-X).

## Licence

Code: MIT (`LICENSE`). Results, corpus and documentation: CC BY 4.0.
