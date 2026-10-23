| Bài toán | Thao tác | Rssa (ms) | ssa-rs (ms) | Nhanh hơn |
|---|---|---:|---:|---:|
| 2D 100x100 circle(10) | decompose | 140 | 30.7 | 4.6× |
|  | reconstruct | 12.0 | 0.75 | 16.1× |
| 2D 100x100 L=20x20 circular | decompose | 230 | 65.0 | 3.5× |
|  | reconstruct | 10.5 | 0.42 | 24.9× |
| 3D 30^3 L=8^3 | decompose | 50.0 | 43.4 | 1.2× |
|  | reconstruct | 10.1 | 0.98 | 10.3× |
|  | esprit.memp | 1.80 | 0.12 | 15.2× |
| 1D circular N=5000 L=1000 | decompose | 10.5 | 7.18 | 1.5× |
|  | rforecast | 12.7 | 0.49 | 25.9× |
| MSSA list 3000+2500 L=500 | decompose | 14.5 | 10.5 | 1.4× |
|  | rforecast.row | 17.3 | 1.10 | 15.6× |
|  | vforecast.col | 5.90 | 0.32 | 18.3× |
| 2D NA 100x100 L=15x15 | decompose | 76.7 | 21.0 | 3.7× |
|  | igapfill | 690 | 159 | 4.3× |
| Gaps N=2000 L=200 | gapfill.sim | 230 | 43.3 | 5.3× |
| CSSA N=2000 L=500 | decompose | 840 | 198 | 4.2× |
|  | esprit | 0.70 | 0.05 | 14.1× |
|  | vforecast | 6.55 | 0.30 | 21.6× |
| hmatr N=500 L=20 | hmatr | 730 | 7.29 | 100.2× |
| PSSA N=2000 L=500 | vforecast | 4.40 | 0.22 | 20.3× |
|  | rforecast | 7.90 | 0.23 | 33.7× |
| W-OSSA N=2000 L=500 | decompose | 1,030 | 139 | 7.4× |
|  | rforecast | 2.73 | 0.31 | 8.7× |
|  | vforecast | 2.55 | 0.15 | 17.2× |
| CSSA N=2000 L=500 | reconstruct | 8.18 | 0.21 | 38.4× |

| Bài toán (tổng các thao tác) | Rssa (ms) | ssa-rs (ms) | Nhanh hơn |
|---|---:|---:|---:|
| 2D 100x100 circle(10) | 152 | 31.5 | 4.8× |
| 2D 100x100 L=20x20 circular | 240 | 65.5 | 3.7× |
| 3D 30^3 L=8^3 | 61.9 | 44.5 | 1.4× |
| 1D circular N=5000 L=1000 | 23.2 | 7.67 | 3.0× |
| MSSA list 3000+2500 L=500 | 37.7 | 11.9 | 3.2× |
| 2D NA 100x100 L=15x15 | 767 | 180 | 4.3× |
| Gaps N=2000 L=200 | 230 | 43.3 | 5.3× |
| CSSA N=2000 L=500 | 855 | 199 | 4.3× |
| hmatr N=500 L=20 | 730 | 7.29 | 100.2× |
| PSSA N=2000 L=500 | 12.3 | 0.45 | 27.3× |
| W-OSSA N=2000 L=500 | 1,035 | 139 | 7.4× |
