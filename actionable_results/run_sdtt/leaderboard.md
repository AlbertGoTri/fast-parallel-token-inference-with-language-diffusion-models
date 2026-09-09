# Nested Distillation Leaderboard


| Round | Steps | Promptfoo % | Perplexity | Cache(s) | Train(s) | Eval(s) | AvgGen(ms) | MedGen(ms) | P95Gen(ms) | Speedup | Pass | Timestamp |
|-------|-------|-------------|------------|----------|----------|---------|------------|------------|------------|---------|------|-----------|
| 0 | 128 | 79.6% | 13.00 | 0.0 | 0.0 | 420.6 | 17279 | 17234 | 17620 |  | ✓ | 2026-09-08T19:30:28 |
| 1 | 64 | 79.6% | 12.97 | 643.7 | 348.4 | 505.5 | 8848 | 8803 | 9119 | 1.95x | ✓ | 2026-09-08T19:55:26 |
| 2 | 32 | 79.6% | 17.09 | 656.1 | 346.8 | 514.3 | 4466 | 4378 | 4879 | 1.98x | ✓ | 2026-09-08T20:20:44 |
| 3 | 16 | 70.4% | 26.08 | 524.8 | 346.8 | 786.8 | 2350 | 2299 | 2655 | 1.90x | ✓ | 2026-09-08T20:48:23 |
| 4 | 8 | 59.3% | 63.63 | 459.3 | 345.9 | 1088.8 | 1372 | 1393 | 1568 | 1.71x | ✓ | 2026-09-08T21:19:57 |
| 5 | 4 | 61.1% | 189.60 | 447.1 | 346.5 | 1165.5 | 862 | 829 | 1023 | 1.59x | ✓ | 2026-09-08T21:52:37 |
| 6 | 2 | 42.6% | 414.11 | 412.0 | 345.2 | 1102.6 | 620 | 611 | 785 | 1.39x | ✓ | 2026-09-08T22:23:37 |
| 7 | 1 | 38.9% | 456.03 | 396.8 | 344.7 | 1119.6 | 445 | 427 | 593 | 1.39x | ✓ | 2026-09-08T22:54:39 |

## Summary

- Student 128 steps: promptfoo assertion 79.6%, perplexity 13.0, avg_gen=17279ms, med_gen=17234ms, p95=17620ms, n=12
- Student 64 steps: promptfoo assertion 79.6%, perplexity 13.0, avg_gen=8848ms, med_gen=8803ms, p95=9119ms, n=12, avg_speedup=1.95x, med_speedup=1.96x
- Student 32 steps: promptfoo assertion 79.6%, perplexity 17.1, avg_gen=4466ms, med_gen=4378ms, p95=4879ms, n=12, avg_speedup=1.98x, med_speedup=2.02x
- Student 16 steps: promptfoo assertion 70.4%, perplexity 26.1, avg_gen=2350ms, med_gen=2299ms, p95=2655ms, n=12, avg_speedup=1.90x, med_speedup=1.94x
- Student 8 steps: promptfoo assertion 59.3%, perplexity 63.6, avg_gen=1372ms, med_gen=1393ms, p95=1568ms, n=12, avg_speedup=1.71x, med_speedup=1.69x
- Student 4 steps: promptfoo assertion 61.1%, perplexity 189.6, avg_gen=862ms, med_gen=829ms, p95=1023ms, n=12, avg_speedup=1.59x, med_speedup=1.66x
- Student 2 steps: promptfoo assertion 42.6%, perplexity 414.1, avg_gen=620ms, med_gen=611ms, p95=785ms, n=12, avg_speedup=1.39x, med_speedup=1.41x
- Student 1 steps: promptfoo assertion 38.9%, perplexity 456.0, avg_gen=445ms, med_gen=427ms, p95=593ms, n=12, avg_speedup=1.39x, med_speedup=1.45x