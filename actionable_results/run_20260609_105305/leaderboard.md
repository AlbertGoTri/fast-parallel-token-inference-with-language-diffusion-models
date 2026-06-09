# Nested Distillation Leaderboard


| Round | Steps | Promptfoo % | Perplexity | Cache(s) | Train(s) | Eval(s) | AvgGen(ms) | MedGen(ms) | P95Gen(ms) | Speedup | Pass | Timestamp |
|-------|-------|-------------|------------|----------|----------|---------|------------|------------|------------|---------|------|-----------|
| 0 | 128 | 74.1% | 13.00 | 0.0 | 0.0 | 780.3 | 17292 | 17266 | 17534 |  | ✓ | 2026-06-09T11:06:05 |
| 1 | 64 | 74.1% | 12.97 | 910.5 | 136.4 | 641.7 | 8934 | 8838 | 9276 | 1.94x | ✓ | 2026-06-09T11:34:15 |
| 2 | 32 | 75.9% | 15.93 | 650.8 | 116.7 | 1240.8 | 4766 | 4703 | 5294 | 1.87x | ✓ | 2026-06-09T12:07:43 |
| 3 | 16 | 68.5% | 24.86 | 517.8 | 90.9 | 1174.3 | 2514 | 2502 | 2681 | 1.90x | ✓ | 2026-06-09T12:37:27 |
| 4 | 8 | 63.0% | 68.44 | 452.9 | 85.6 | 1219.6 | 1412 | 1393 | 1522 | 1.78x | ✓ | 2026-06-09T13:06:46 |
| 5 | 4 | 59.3% | 79.18 | 422.8 | 85.9 | 1133.2 | 855 | 833 | 1006 | 1.65x | ✓ | 2026-06-09T13:34:08 |
| 6 | 2 | 50.0% | 161.93 | 404.6 | 86.3 | 1048.3 | 562 | 596 | 685 | 1.52x | ✓ | 2026-06-09T13:59:48 |
| 7 | 1 | 46.3% | 353.26 | 399.1 | 90.4 | 1103.0 | 468 | 476 | 558 | 1.20x | ✓ | 2026-06-09T14:26:21 |

## Summary

- Student 128 steps: promptfoo assertion 74.1%, perplexity 13.0, avg_gen=17292ms, med_gen=17266ms, p95=17534ms, n=12
- Student 64 steps: promptfoo assertion 74.1%, perplexity 13.0, avg_gen=8934ms, med_gen=8838ms, p95=9276ms, n=12, avg_speedup=1.94x, med_speedup=1.96x
- Student 32 steps: promptfoo assertion 75.9%, perplexity 15.9, avg_gen=4766ms, med_gen=4703ms, p95=5294ms, n=12, avg_speedup=1.87x, med_speedup=1.90x
- Student 16 steps: promptfoo assertion 68.5%, perplexity 24.9, avg_gen=2514ms, med_gen=2502ms, p95=2681ms, n=12, avg_speedup=1.90x, med_speedup=1.91x
- Student 8 steps: promptfoo assertion 63.0%, perplexity 68.4, avg_gen=1412ms, med_gen=1393ms, p95=1522ms, n=12, avg_speedup=1.78x, med_speedup=1.80x
- Student 4 steps: promptfoo assertion 59.3%, perplexity 79.2, avg_gen=855ms, med_gen=833ms, p95=1006ms, n=12, avg_speedup=1.65x, med_speedup=1.69x
- Student 2 steps: promptfoo assertion 50.0%, perplexity 161.9, avg_gen=562ms, med_gen=596ms, p95=685ms, n=12, avg_speedup=1.52x, med_speedup=1.43x
- Student 1 steps: promptfoo assertion 46.3%, perplexity 353.3, avg_gen=468ms, med_gen=476ms, p95=558ms, n=12, avg_speedup=1.20x, med_speedup=1.18x