# CISE Release Summary

**Constraint-Induced Structure Explorer v1.0.0**

---

> **Note**: This is a constraint experiment exploring how penalty functions
> affect ensemble distributions. It does not claim physical truth.

---

## Configuration

- **Seed**: 1337
- **Samples**: 8000
- **Vector dim**: 32
- **Matrix size**: 6x6
- **Beta sweep**: [0.0, 0.25, 0.5, 1.0]

## Key Results

### vector_gaussian_beta0.25

- **ESS**: 129.9 (1.6%)
- **Norm shift**: -1.375
- **Gini shift**: +0.0003

*Low ESS indicates concentration of measure; interpret with caution.*

### vector_gaussian_beta0.50

- **ESS**: 18.4 (0.2%)
- **Norm shift**: -1.748
- **Gini shift**: +0.0052

*Low ESS indicates concentration of measure; interpret with caution.*

### vector_gaussian_beta1.00

- **ESS**: 5.0 (0.1%)
- **Norm shift**: -1.966
- **Gini shift**: +0.0183

*Low ESS indicates concentration of measure; interpret with caution.*

### matrix_gaussian_beta0.25

- **ESS**: 5158.6 (64.5%)
- **Norm shift**: -0.347
- **Gini shift**: +0.0040
- **Rank proxy shift**: -0.078

### matrix_gaussian_beta0.50

- **ESS**: 2389.8 (29.9%)
- **Norm shift**: -0.583
- **Gini shift**: +0.0069
- **Rank proxy shift**: -0.148

### matrix_gaussian_beta1.00

- **ESS**: 515.9 (6.4%)
- **Norm shift**: -0.883
- **Gini shift**: +0.0102
- **Rank proxy shift**: -0.275

*Low ESS indicates concentration of measure; interpret with caution.*

## Anti-Dismissal Control

**Norm-Matched Baseline Control**: Addresses the critique that observed
structure changes are merely due to norm shrinkage.

### Vector Control

- Norm (constrained): 3.645
- Norm (matched): 3.643
- Participation ratio delta: -5.502
- Gini delta (beyond norm): +0.0194

*Structure differences persist after matching norm distributions.*

### Matrix Control

- Norm (constrained): 5.083
- Norm (matched): 5.086
- Participation ratio delta: -0.230
- Gini delta (beyond norm): +0.0107

*Structure differences persist after matching norm distributions.*

## Figures

### Release Figures

- [matrix_energy_histogram.png](figures_release/matrix_energy_histogram.png)
- [matrix_ess_vs_beta.png](figures_release/matrix_ess_vs_beta.png)
- [matrix_gini_distribution.png](figures_release/matrix_gini_distribution.png)
- [matrix_hierarchy.png](figures_release/matrix_hierarchy.png)
- [matrix_norm_distribution.png](figures_release/matrix_norm_distribution.png)
- [matrix_pca_scatter.png](figures_release/matrix_pca_scatter.png)
- [matrix_pca_variance.png](figures_release/matrix_pca_variance.png)
- [matrix_sv_spectrum.png](figures_release/matrix_sv_spectrum.png)
- [vector_energy_histogram.png](figures_release/vector_energy_histogram.png)
- [vector_ess_vs_beta.png](figures_release/vector_ess_vs_beta.png)
- [vector_gini_distribution.png](figures_release/vector_gini_distribution.png)
- [vector_hierarchy.png](figures_release/vector_hierarchy.png)
- [vector_norm_distribution.png](figures_release/vector_norm_distribution.png)
- [vector_pca_scatter.png](figures_release/vector_pca_scatter.png)
- [vector_pca_variance.png](figures_release/vector_pca_variance.png)

### Control Figures

- [matrix_control_pca_scatter.png](figures_control/matrix_control_pca_scatter.png)
- [matrix_control_pca_variance.png](figures_control/matrix_control_pca_variance.png)
- [vector_control_pca_scatter.png](figures_control/vector_control_pca_scatter.png)
- [vector_control_pca_variance.png](figures_control/vector_control_pca_variance.png)

---

## Interpretation Guidelines

- Constraints induce distributional shifts toward lower-energy configurations
- Structure changes (dimensional concentration, rank reduction) emerge from constraint geometry
- The norm-matched control shows these effects persist beyond simple magnitude reduction
- Low ESS at high beta indicates measure concentration; results should be interpreted carefully

**This is a constraint experiment, not a physics claim.**
