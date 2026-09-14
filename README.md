# Investigating Occupational Gender Bias in CLIP
## The Dominant Role of Embedding Geometry

[![Paper](https://img.shields.io/badge/Paper-OpenReview-8A2BE2)](https://openreview.net/forum?id=ve0WcrVkcE)
![Model](https://img.shields.io/badge/Model-CLIP%20ViT--B%2F32-blue)
![Dataset](https://img.shields.io/badge/Dataset-FairFace-green)
![Topic](https://img.shields.io/badge/Topic-Responsible%20Multimodal%20AI-orange)

Official research code for a controlled study of **occupational gender association gaps in CLIP**. The project asks a measurement-focused question:

> **When CLIP is evaluated for occupational gender bias, which experimental choice changes the measured gap the most?**

We compare three factors under a controlled FairFace protocol: **evaluation-pool gender composition**, **prompt formulation**, and **embedding geometry**. The associated manuscript is available on [OpenReview](https://openreview.net/forum?id=ve0WcrVkcE).

---

## Main finding

Across the factorial experiment, **embedding geometry is the dominant source of variation in the measured absolute gender association gap**.

| Factor | Partial η² | Interpretation |
| --- | ---: | --- |
| **Embedding method** | **0.6447** | Dominant effect |
| Prompt type | 0.0660 | Smaller but detectable effect |
| Data setting | 0.0065 | Limited effect under this metric |

Matched comparisons across 180 conditions show the following mean absolute gaps:

| Embedding method | Mean absolute gap |
| --- | ---: |
| **Gender-direction removed** | **0.00206** |
| Standard cosine | 0.01038 |
| Mean-centered | 0.02269 |

The projection of occupation text embeddings onto the estimated gender direction is also almost perfectly aligned with the signed gender gap in the evaluated conditions (**Pearson r = 0.99985**).

> These results do **not** imply that removing a linear gender direction makes CLIP universally fair. They show that representation geometry strongly influences this specific bias measurement.

---

## Experimental design

### Model and data
- **Vision-language model:** OpenAI CLIP ViT-B/32
- **Dataset:** FairFace
- **Age group:** 30–39
- **Controlled base pool:** 4,200 images
- **Sampling control:** 7 race groups × 2 genders × 300 images
- **Occupation probes:** 20

Race is used only as a sampling-control variable in this study, not as the research target.

### Full-factorial factors
- **3 evaluation-pool gender settings:** balanced 50/50, male-skewed 80/20, female-skewed 20/80
- **3 prompt formulations:** short, medium, long
- **3 embedding methods:** standard cosine, mean-centered, gender-direction removed
- **20 occupations**

This yields **540 occupation-level experimental conditions**.

### Primary metric

```text
signed_gap = mean(similarity_male) - mean(similarity_female)
absolute_bias = |signed_gap|
```

Lower `absolute_bias` means a smaller measured male/female association gap under the selected evaluation geometry.

---

## Embedding-space interventions

### Standard cosine
```text
sim(x, t) = normalize(x) · normalize(t)
```

### Mean-centered geometry
```text
x' = x - μ
t' = t - μ
```

### Gender-direction removal
```text
g = mean(image_male) - mean(image_female)
x' = x - proj_g(x)
t' = t - proj_g(t)
```

The reusable implementation is in [`src/clip_bias_geometry/geometry.py`](src/clip_bias_geometry/geometry.py).

---

## Repository structure

```text
.
├── README.md
├── REPRODUCIBILITY.md
├── NOTICE.md
├── requirements.txt
├── scripts/
│   ├── run_experiment.py
│   └── analyze_results.py
├── src/
│   └── clip_bias_geometry/
│       ├── geometry.py
│       ├── metrics.py
│       ├── prompts.py
│       └── statistics.py
└── results/
    └── key_results.csv
```

The repository is intentionally organized as a **paper companion artifact** rather than a course-style notebook dump: the main experimental pipeline is executable from `scripts/`, while core transformations and statistics are factored into reusable modules.

---

## Reproduce the experiment

### 1. Clone

```bash
git clone https://github.com/ERRT-ai/clip-gender-bias-geometry.git
cd clip-gender-bias-geometry
```

### 2. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the factorial experiment

```bash
python scripts/run_experiment.py
```

The script downloads the FairFace resources used by the original experiment, builds the controlled evaluation pools, caches CLIP image embeddings, and writes the 540-row factorial result table under `artifacts/results/tables/`.

### 4. Run the statistical analysis

```bash
python scripts/analyze_results.py
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the exact construction rules and expected key values.

---

## Statistical analysis

The study uses:
- Type-II factorial ANOVA and variance decomposition
- partial η² effect sizes
- paired t-tests
- Wilcoxon signed-rank tests
- bootstrap confidence intervals
- Pearson and Spearman correlations

The main ANOVA result for embedding method is **F = 455.50**, **partial η² = 0.6447**, with **p = 1.55 × 10⁻¹¹³** in the archived experiment.

---

## Responsible interpretation and limitations

This repository studies **measurement sensitivity**, not fairness certification.

- FairFace provides binary gender labels, which do not represent the full spectrum of gender identity.
- Occupational association scores depend on the model, prompts, dataset, similarity function, and evaluation protocol.
- FairFace is an external dataset and is **not redistributed** in this repository.
- A lower value for this metric does not establish absence of other forms of bias or harmful behavior.
- Linear gender-direction removal should therefore be interpreted as an embedding-space intervention used to study the geometry of the measured association signal.

---

## Paper

Associated manuscript: [OpenReview: ve0WcrVkcE](https://openreview.net/forum?id=ve0WcrVkcE)

Formal citation metadata can be added once the final bibliographic record is fixed.

---

## Acknowledgements

This research builds on OpenAI CLIP, FairFace, PyTorch, SciPy, statsmodels, NumPy and pandas. Please follow the original licenses and usage terms of all third-party models and datasets.
