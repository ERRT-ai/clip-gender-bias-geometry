# CLIP Gender Bias Geometry

Research code for the OpenReview paper: https://openreview.net/forum?id=ve0WcrVkcE

This project studies how evaluation-pool composition, prompt formulation, and embedding geometry affect measured occupational gender association gaps in CLIP.

## Main finding

Embedding geometry is the dominant factor in the measured absolute gender association gap under the controlled FairFace evaluation protocol.

## Core setup

- Model: OpenAI CLIP ViT-B/32
- Dataset: FairFace
- 20 occupation probes
- 3 gender-composition settings
- 3 prompt formulations
- 3 embedding-space methods
- 540 occupation-level experimental conditions

## Key results

| Factor | Partial eta squared |
| --- | ---: |
| Embedding method | 0.6447 |
| Prompt type | 0.0660 |
| Data setting | 0.0065 |

Matched comparisons show a mean absolute gap of 0.00206 after gender-direction removal, compared with 0.01038 for standard cosine and 0.02269 after mean-centering.

## Repository structure

```text
notebooks/   End-to-end experiment
src/         Reusable analysis utilities
results/     Key reported numerical results
assets/      Figures used by the README
```

## Reproduction

```bash
git clone https://github.com/ERRT-ai/clip-gender-bias-geometry.git
cd clip-gender-bias-geometry
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/clip_gender_bias_geometry.ipynb
```

## Research scope

This is a measurement study, not a fairness certification framework. Reducing the reported absolute association gap does not imply that CLIP is universally fair or free of demographic bias.
