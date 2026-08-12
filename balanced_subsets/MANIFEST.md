# Balanced shareable subsets of soft/hard v2

Balance = **equal episodes per shape** (whole episodes, seed-fixed).
Rows are **interleaved round-robin** across shapes, so any front-prefix stays balanced.
Selection seed = 0. Shapes = circle/pentagon/square/triangle (star is untrained, absent).

| file | eps/shape | shapes | episodes | rows |
|---|---|---|---|---|
| soft_v2_100k_balanced.csv.gz | 7 | circle:7 pentagon:7 square:7 triangle:7 | 28 | 100269 |
| soft_v2_500k_balanced.csv.gz | 35 | circle:35 pentagon:35 square:35 triangle:35 | 140 | 498396 |
| hard_v2_100k_balanced.csv.gz | 7 | circle:7 pentagon:7 square:7 triangle:7 | 28 | 100269 |
| hard_v2_500k_balanced.csv.gz | 35 | circle:35 pentagon:35 square:35 triangle:35 | 140 | 498396 |
