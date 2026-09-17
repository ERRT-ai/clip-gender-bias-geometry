# Experimental Results

This document records the main numerical results produced by the reference notebook. It is intended to make the repository auditable without requiring readers to inspect every notebook output.

## Primary outcome

The main outcome is the absolute occupational gender association gap:

```text
absolute_bias = |mean(similarity_male) - mean(similarity_female)|
```

The factorial experiment contains 540 occupation-level conditions: 3 data settings × 3 prompt types × 3 embedding methods × 20 occupations.

## Variance decomposition

The factorial ANOVA shows that embedding geometry explains substantially more variation in measured absolute bias than either prompt wording or evaluation-pool gender composition.

| Term | F | p-value | Partial eta squared |
| --- | ---: | ---: | ---: |
| Embedding method | 455.503457 | 1.550304e-113 | **0.644729** |
| Prompt type | 17.736567 | 3.606503e-08 | 0.066000 |
| Prompt × embedding | 4.630013 | 1.116358e-03 | 0.035580 |
| Data × embedding | 1.757998 | 1.360428e-01 | 0.013814 |
| Data setting | 1.631674 | 1.966375e-01 | 0.006459 |
| Data × prompt | 0.002647 | 9.999860e-01 | 0.000021 |

The main result is therefore about **measurement sensitivity to representation geometry**, not a universal ranking of fairness interventions.

## Matched embedding-method comparisons

Each method is compared on the same 180 data-setting × prompt × occupation conditions.

| Method A | Method B | Mean A | Mean B | Difference B-A | Paired t-test p | Wilcoxon p |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Standard cosine | Gender-direction removed | 0.010381 | **0.002056** | -0.008324 | 2.471971e-32 | 3.494327e-26 |
| Standard cosine | Mean-centered | 0.010381 | 0.022689 | +0.012309 | 1.079277e-40 | 4.008186e-26 |
| Gender-direction removed | Mean-centered | 0.002056 | 0.022689 | +0.020633 | 3.270598e-46 | 1.145434e-30 |

Under this metric, removing the estimated linear gender direction produces the smallest mean absolute association gap, while mean-centering increases the gap relative to standard cosine similarity.

## Mechanism analysis

For the balanced evaluation pool with standard cosine similarity, the projection of occupation text embeddings onto the estimated image-derived gender direction is strongly aligned with the signed male-minus-female association gap.

| Comparison | N | Pearson r | Pearson p | Spearman rho | Spearman p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text projection vs. signed gender gap | 60 | **0.999851** | 5.687044e-104 | 0.999166 | 2.822063e-82 |

This mechanism result supports the interpretation that the evaluated association signal is tightly coupled to the geometry of the CLIP representation space.

## Interpretation boundaries

These results should not be interpreted as evidence that projecting out one direction makes CLIP generally fair. The experiment measures one operationalized association gap on one controlled FairFace protocol, with binary gender labels and a fixed set of occupations. Other datasets, demographic definitions, prompt sets, downstream tasks, and fairness criteria may produce different conclusions.
