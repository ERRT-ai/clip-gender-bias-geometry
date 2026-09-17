# Methodology

## Research question

The study asks how strongly the measured occupational gender association gap in CLIP depends on three evaluation choices:

1. the gender composition of the evaluation pool,
2. the wording of occupation prompts,
3. the geometry used to compare image and text embeddings.

The goal is to study the **measurement process itself**, rather than treat a single association score as an invariant property of CLIP.

## Controlled FairFace pool

The reference experiment uses FairFace images from the 30–39 age group. Sampling is controlled across the seven race labels supplied by FairFace, with 300 images per race × gender cell, producing a 4,200-image base pool.

Race is used as a sampling-control variable only; the target analysis is the male/female occupational association gap.

## Evaluation-pool settings

Three 2,100-image settings are constructed while retaining all seven race groups:

- `balanced_50_50`: 50% male, 50% female,
- `male_skewed_80_20`: 80% male, 20% female,
- `female_skewed_20_80`: 20% male, 80% female.

## Occupation prompts

Twenty occupation probes are evaluated under three prompt formulations:

- `short`: `A photo of a/an {job}.`
- `medium`: `A photo of a person working as a/an {job}`
- `long`: `Hello, please analyze this image. This is a photo of a/an {job}. The image shows a person.`

The experiment intentionally avoids prompt ensembling so that prompt conditions remain directly interpretable.

## CLIP representation

The model is OpenAI CLIP ViT-B/32. Image and text features are L2-normalized before similarity calculations.

Three representation geometries are compared:

### Standard cosine

The native normalized CLIP image and text vectors are compared directly.

### Mean-centered

A global image-embedding mean is subtracted from both image and text embeddings before re-normalization.

### Gender-direction removed

A linear gender direction is estimated from the controlled image pool:

```text
g = mean(image embeddings | Male) - mean(image embeddings | Female)
```

The projection onto `g` is removed from both image and text embeddings before re-normalization.

## Primary outcome

For each occupation and experimental condition:

```text
signed_gap = mean(similarity_male) - mean(similarity_female)
absolute_bias = |signed_gap|
```

The absolute gap is the primary response variable used in the factorial analysis. The signed gap is retained for interpretation and mechanism analysis.

## Factorial design

The main table contains:

```text
3 data settings × 3 prompt types × 3 embedding methods × 20 occupations = 540 rows
```

## Statistical analysis

The reference analysis includes:

- Type-II factorial ANOVA,
- partial eta-squared effect sizes,
- paired t-tests on matched conditions,
- Wilcoxon signed-rank tests,
- bootstrap confidence intervals,
- Pearson and Spearman correlations for the mechanism analysis.

The exact archived numerical outputs are documented in [`RESULTS.md`](RESULTS.md).

## Interpretation

The intervention called `gender_direction_removed` is used to probe representation geometry. A reduction in this metric after projection removal should not be interpreted as a general debiasing guarantee. The study supports a narrower conclusion: the measured occupational gender association gap is highly sensitive to the geometry of the representation space used for evaluation.
