# CISE Launch Kit

Materials for announcing CISE v1.0.0 on social media and video.

---

## X/Twitter Thread (3-6 posts)

### Post 1 (Hook)
```
Constraints → Structure

I built a tool to test a simple question: do coherence constraints (smoothness, sparsity, low-rank) induce non-trivial structure in random ensembles?

Yes. Measurably. And it persists after controlling for norm shrinkage.

CISE is open source. 🧵
```

### Post 2 (Evidence)
```
Key findings from CISE v1.0:

• Norm distributions shift (constraints favor smaller magnitudes)
• PCA variance concentrates in fewer dimensions
• Matrix singular values show rank reduction
• Gini coefficient increases (sparse structure emerges)

All from simple penalty functions.
```

### Post 3 (Control)
```
"But isn't this just because you're shrinking norms?"

Good question. I built a control: a norm-matched baseline with the same norm distribution as constrained samples.

Structure differences persist. Constraint geometry matters, not just magnitude.

[Figure: control_pca_variance.png]
```

### Post 4 (Figures)
```
Visual evidence:

1️⃣ ESS drops with constraint strength (measure concentration)
2️⃣ PCA scatter shows clustering in constrained samples
3️⃣ Singular value spectrum shows rank suppression
4️⃣ Hierarchy curves show steeper magnitude decay

[Figure grid: 4 key figures]
```

### Post 5 (Invitation)
```
CISE is:
✓ Reproducible (fixed seed, <2 min on CPU)
✓ Extensible (add your own constraints)
✓ Neutral (no physics claims)

Fork it. Add constraints. Find counterexamples.

https://github.com/vfd-org/constraint-induced-structure
```

### Post 6 (Disclaimer)
```
To be clear: CISE is a constraint experiment, not a physics model.

It shows that constraints → structure in statistical ensembles.

It does NOT claim this explains any natural phenomenon.

Reproducible, falsifiable, open to critique.
```

---

## Video Script (90-120 seconds)

### Opening (0-15s)
```
[Screen: Terminal]
"What happens when you apply simple constraints to random samples?
Let me show you."

[Type command]
python scripts/run_release.py
```

### Demo (15-45s)
```
[Screen: Output appearing]
"CISE generates 8000 random vectors and matrices, applies constraints
like smoothness and low-rank via energy-based reweighting, and measures
what changes."

[Screen: figures_release folder opening]
"It produces these canonical figures in under 2 minutes."
```

### Key Figures (45-75s)
```
[Show figure grid: 2x3]
"Look at the ESS curve—effective sample size drops as constraint
strength increases. Measure concentration.

The PCA variance shows dimensional concentration. Fewer dimensions
capture more variance.

The singular value spectrum shows rank suppression. Constraints induce
low-rank structure."
```

### Control (75-95s)
```
[Show control figures]
"Now the important part. Someone might say 'this is just norm shrinkage.'

So I built a control—a norm-matched baseline with the same norm
distribution. Structure differences persist.

Constraint geometry matters."
```

### Close (95-120s)
```
[Screen: README]
"CISE is open source, reproducible, and extensible.

Fork it. Add your own constraints. Find counterexamples.

This is a constraint experiment, not a physics claim.
Reproducible, falsifiable, open."

[Show URL]
https://github.com/vfd-org/constraint-induced-structure
```

---

## FAQ

### "Isn't this trivial because of shrinkage?"

Point to the norm-matched baseline control. After matching norm distributions, structure differences persist. The participation ratio delta and Gini delta are non-zero. Constraint geometry—not just magnitude reduction—drives the changes.

See: `outputs/release_run/figures_control/`

### "Does this claim new physics?"

No. CISE is a computational experiment studying how constraints affect statistical ensembles. It makes no claims about physical reality, natural phenomena, or experimental predictions.

The README explicitly states what it is and isn't.

### "What's the point then?"

Demonstrating that simple, generic constraints induce measurable structure is itself interesting. The framework is:
- A testbed for studying constraint effects
- A benchmark for comparing constraint types
- A template for extension and counterexample testing

### "What's next?"

Potential extensions:
- Add more constraint types
- Test on different base distributions
- Explore higher-dimensional samples
- Benchmark against alternative reweighting methods
- Community contributions and counterexamples

### "How do I cite this?"

See `CITATION.cff` or use:
```bibtex
@software{cise2024,
  author = {Smart, Lee},
  title = {CISE: Constraint-Induced Structure Explorer},
  year = {2024},
  version = {1.0.0},
  url = {https://github.com/vfd-org/constraint-induced-structure}
}
```

---

## Assets Checklist

Before launch, ensure:

- [ ] README renders correctly on GitHub
- [ ] Release figures are generated and look clean
- [ ] Control figures show meaningful difference
- [ ] CITATION.cff is valid
- [ ] License file present
- [x] Repository URL set to https://github.com/vfd-org/constraint-induced-structure
- [ ] One-page note compiles to PDF
- [ ] Video/screenshots captured

---

## Hashtags

For discoverability:
```
#OpenSource #Statistics #MachineLearning #DataScience #ComputationalScience #Reproducibility
```
