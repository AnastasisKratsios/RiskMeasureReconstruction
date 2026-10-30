# Reference data

These are inspectable results, not inputs required to run an experiment.
Always write new runs under a fresh `results/` directory. Do not resume into
these reference directories; per-draw checkpoints are deliberately omitted.

| Path | Provenance |
| --- | --- |
| `paper_reported.csv` | Values transcribed from the supplied manuscript at its displayed precision |
| `paper_source.json` | Manuscript title, author names, table labels, and source-text SHA-256 |
| `main_run/` | Fresh run of this repository: all three models, four families, 30 draws each |
| `ablation_run/` | Fresh, explicitly configured gamma ablation: 30 paired draws at 11 gamma values |

The main run includes 360 per-draw records and 43,200 predictions. Its three
LaTeX tables agree with the supplied paper at the displayed precision. The
comparison script checks 72 reported statistics. For the finite case it uses
an absolute roundoff-scale threshold, since relative differences between
approximately zero MSEs are not meaningful.

The ablation includes 330 conic fits and 36,300 prediction rows, along with
both figures in PDF and PNG. It is **not** a recovery of the draft's unavailable
ablation figure data. Its settings, resulting error levels, and the manuscript
changes needed to use these figures are described in
[`docs/paper_alignment.md`](../docs/paper_alignment.md).

Inspect the data or regenerate the tables:

```bash
risk-reconstruct validate --input reference/main_run
risk-reconstruct validate --input reference/ablation_run
python scripts/compare_paper.py --input reference/main_run
```

The configurations record exact arrays, source hashes, and runtime versions.
Timing fields describe the verification host and are not performance promises.
CSV files use full stored floating-point precision; LaTeX is rounded only for
presentation. `validation.json` describes the numerical checks, which do not
constitute rigorous interval arithmetic for the continuous-law integration.

