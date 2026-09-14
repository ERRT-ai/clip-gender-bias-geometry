# Reproducibility Guide

## Environment

Recommended setup:
- Python 3.10+
- CUDA-capable GPU
- OpenAI CLIP ViT-B/32

Install dependencies with:

```bash
pip install -r requirements.txt
```

For an archival release, record the exact Python, PyTorch, CUDA and CLIP commit versions used in the final run.

## Dataset

The experiment uses the FairFace training split. The pipeline:
1. filters to age group `30-39`;
2. groups images by `race × gender`;
3. samples 300 images per group with a fixed seed;
4. constructs a controlled base pool of 4,200 images.

FairFace is not redistributed in this repository. Users should follow the original dataset terms and license.

## Experimental factors

### Evaluation-pool gender composition
Each evaluation pool contains 2,100 images while preserving coverage across all seven FairFace race groups.

- `balanced_50_50`
- `male_skewed_80_20`
- `female_skewed_20_80`

### Prompt formulation
Three prompt conditions are used:
- `short`
- `medium`
- `long`

### Embedding geometry
Three representation-space conditions are evaluated:
- `standard_cosine`
- `mean_centered`
- `gender_direction_removed`

## Main metric

For every data-setting × prompt × embedding-method × occupation combination:

```text
signed_gap = mean(similarity_male) - mean(similarity_female)
absolute_bias = |signed_gap|
```

The complete factorial table contains 540 occupation-level rows.

## Statistical analysis

The main analysis uses Type-II ANOVA to decompose variation in `absolute_bias` across data setting, prompt type, embedding method, occupation, and selected two-way interactions. Partial eta squared is reported as the primary effect-size measure.

Matched embedding-method comparisons additionally use paired t-tests, Wilcoxon signed-rank tests and bootstrap confidence intervals.

## Expected key values

| Quantity | Archived value |
| --- | ---: |
| Embedding-method partial eta squared | 0.644729 |
| Prompt-type partial eta squared | 0.066000 |
| Data-setting partial eta squared | 0.006459 |
| Standard-cosine mean absolute gap | 0.010381 |
| Direction-removed mean absolute gap | 0.002056 |
| Mean-centered mean absolute gap | 0.022689 |
| Projection vs signed-gap Pearson r | 0.999851 |

Small numerical differences may occur across hardware and dependency versions.
