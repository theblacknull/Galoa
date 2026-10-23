# ssa-rs: báo cáo hiệu năng và độ chính xác

Thư viện SSA viết bằng Rust (`rust/`), so sánh với hai bản gốc:

- **py-ssa-lib 2.0.2** (Python): đã port toàn bộ, gọi từ Python với cùng API.
- **Rssa 1.1** (R, lõi C/Fortran của nhóm Golyandina): đã port các phương pháp mà bản Python không có.

Máy đo: Ryzen 7 5800H (8 nhân), 32 GB RAM, Windows 11. Mỗi phép đo lấy trung vị nhiều lần chạy. Hai bên dùng cùng dữ liệu và cùng tác vụ.

## 1. So với py-ssa-lib

Một quy trình gồm: fit, tái tạo 10 thành phần, W-correlation, L-/K-forecast 50 bước, ESPRIT. Thời gian Rust đã tính cả chi phí gọi từ Python.

| Bài toán | py-ssa-lib | ssa-rs | Nhanh hơn | Bộ nhớ (Python → Rust) |
|---|---:|---:|---:|---:|
| SSA N=200, L=100 | 98 ms | 2.7 ms | 37× | 155 → 72 MB |
| SSA N=500, L=250 | 706 ms | 19 ms | 37× | 496 → 78 MB |
| SSA N=1000, L=500 | 5.35 s | 93 ms | 57× | 3014 → 95 MB |
| SSA N=2000, L=200 | 3.07 s | 40 ms | 77× | 1797 → 87 MB |
| SSA N=5000, L=100 | 3.01 s | 36 ms | 85× | 1277 → 90 MB |
| SSA N=10000, L=100 | 5.92 s | 64 ms | 93× | 2442 → 110 MB |
| MSSA 10 chuỗi, N=300 | 1.37 s | 13 ms | 108× | 603 → 80 MB |
| MSSA 5 chuỗi, N=500 | 2.64 s | 27 ms | 99× | 1524 → 86 MB |
| SSA N=4000, L=2000 | hết RAM | 2.9 s | – | – |
| SSA N=20000, L=1000 | hết RAM | 2.2 s | – | – |

Với tuỳ chọn `EigenSVD`, phần lớn các ca trên còn nhanh thêm 1,5–2 lần.

Lý do nhanh hơn:

- Không lưu sẵn các ma trận sơ cấp (bản Python giữ d·L·K số thực, chiếm vài GB).
- Hankel hoá bằng tích chập FFT, thay cho vòng lặp Python trên từng đường chéo.
- Tính W-correlation bằng một phép nhân ma trận.

## 2. So với Rssa

Rssa mặc định dùng eigen đầy đủ khi L < 500, và Lanczos (nutrlan) lấy 50 thành phần khi L ≥ 500. ssa-rs dùng cùng quy tắc: SVD đầy đủ khi L < 500, `LanczosSvd` khi L ≥ 500.

| Bài toán | Rssa | ssa-rs | Nhanh hơn |
|---|---:|---:|---:|
| 1D N=1000, L=400 | 146 ms | 51 ms | 2.9× |
| 1D N=10⁴, L=2000 | 167 ms | 48 ms | 3.5× |
| 1D N=10⁵, L=10⁴ | 1.06 s | 0.33 s | 3.2× |
| 1D N=5·10⁵, L=5000 | 4.60 s | 0.83 s | 5.6× |
| MSSA 5×2000, L=500 | 198 ms | 29 ms | 6.9× |
| 2D-SSA 200×200, L=50×50 | 104 ms | 34 ms | 3.0× |
| Toeplitz SSA N=5000, L=1000 | 14.5 ms | 3.6 ms | 4.1× |
| Cadzow, I-OSSA, F-OSSA, E-OSSA, auto-grouping | 245 ms | 36 ms | 6.9× |
| Lấp dữ liệu thiếu (shaped SSA, gapfill, igapfill) | 365 ms | 260 ms | 1.4× |
| Projection SSA N=2000, L=500 | 11 ms | 6.6 ms | 1.7× |
| 2D cửa sổ tròn 100×100, circle(10) | 152 ms | 31 ms | 4.8× |
| 2D topo vòng 100×100, L=20×20 | 240 ms | 65 ms | 3.7× |
| 3D-SSA 30³, L=8³ (kèm ESPRIT memp) | 62 ms | 45 ms | 1.4× |
| 1D topo vòng N=5000, L=1000 | 23 ms | 7.7 ms | 3.0× |
| MSSA hai chuỗi dài khác nhau (3000 + 2500) | 38 ms | 12 ms | 3.2× |
| 2D có dữ liệu thiếu + igapfill | 767 ms | 180 ms | 4.3× |
| gapfill "simultaneous" N=2000 | 230 ms | 43 ms | 5.3× |
| Complex SSA N=2000, L=500 | 855 ms | 199 ms | 4.3× |
| hmatr N=500 | 730 ms | 7.3 ms | 100.2× |
| Dự báo Projection SSA | 12 ms | 0.5 ms | 27.3× |
| W-OSSA N=2000, L=500 (phân rã + dự báo) | 1035 ms | 139 ms | 7.4× |

Ở cấp từng thao tác, ssa-rs nhanh hơn ở tất cả 65 thao tác đã đo:

- Tái tạo nhanh hơn 9–95 lần.
- wcor nhanh hơn 6–56 lần.
- Dự báo nhanh hơn 5–90 lần.
- Phân rã nhanh hơn 1,1–9 lần. Đây là phần Rssa vốn đã tối ưu tốt bằng FFTW và TRLan.

Bảng chi tiết nằm trong `rust/benchmarks/`.

## 3. Độ chính xác

- **Khớp py-ssa-lib:** chạy cùng bộ test gốc trên cả hai thư viện, và đối chiếu từng đại lượng (σ, U/V, tái tạo, W-corr, LRR, dự báo, ESPRIT) trên 8 cấu hình. Sai lệch ≤ 1e-9.
- **Khớp Rssa:** 27 bộ đối chiếu với kết quả do chính Rssa tính, gồm nD-SSA (2D, 3D), cửa sổ tròn và cửa sổ có lỗ, topo vòng (1D, 2D, Toeplitz), trường có NA, MSSA chuỗi khác độ dài (r/v-forecast hai hướng, gapfill, igapfill), PSSA/W-OSSA 2D, ESPRIT nD (diag, memp, hàng/cột), gapfill simultaneous, dự báo PSSA (original, reverse, vforecast), CSSA (ESPRIT, vforecast, Cadzow, wnorm, contributions, gapfill, igapfill), dự báo W-OSSA, hmatr, forecast, tau/cond của I-OSSA, 1D, Toeplitz, MSSA, CSSA, 2D-SSA, PSSA, W-OSSA, nutrlan so với `LanczosSvd`, Cadzow, gapfill, igapfill, I-OSSA, F-OSSA, E-OSSA, owcor, wcor, frobenius.cor, auto-grouping, LRR/roots, r/v-forecast (cả reverse và hai hướng MSSA), ESPRIT LS/TLS, pairs. Sai lệch ≤ 1e-8.
- `LanczosSvd` khớp SVD đầy đủ tới 1e-11. Dù vậy SVD đầy đủ vẫn là mặc định; Lanczos chỉ dùng khi chủ động chọn, với chuỗi dài và khi chỉ cần một số thành phần đầu.
- Bộ test đã được kiểm chứng là bắt được lỗi: khi cấy sai số 1e-6 vào dữ liệu tham chiếu, test báo lỗi đúng chỗ.

Tổng cộng có 68 test Rust và 43 test Python, tất cả đều qua.

## 4. Những gì đã có trong ssa-rs

**Từ py-ssa-lib:** SSA, MSSA có trọng số kênh, FullSVD, RandomizedSVD, 4 chiến lược trọng số, bootstrap, 5 loại biểu đồ, dataset.

**Từ Rssa (đầy đủ):**
- Phân rã: SVD đầy đủ, eigen, Lanczos trên toán tử FFT (thay cho nutrlan/propack/irlba/primme/rspectra, cùng kết quả), randomized SVD (thay cho rsvd), tiếp tục phân rã thêm thành phần.
- Loại SSA: 1D, Toeplitz, MSSA (kể cả chuỗi dài khác nhau và có NA), Complex SSA (kể cả chuỗi có NA), 2D-SSA và nD-SSA với số chiều bất kỳ.
- Hình dạng: cửa sổ có hình bất kỳ (wmask, circle, triangle), trường có dữ liệu thiếu, topo vòng theo từng chiều.
- Projection SSA và W-OSSA cho 1D, 2D, nD, dùng riêng hoặc kết hợp; dự báo r/v cho W-OSSA.
- Lấp khoảng trống: gapfill (sequential và simultaneous), igapfill, cho cả chuỗi thực và chuỗi phức; summarize.gaps, clplot, Cadzow.
- Phân rã xiên: I-OSSA (kèm tau, cond, owcor), F-OSSA/DerivSSA, E-OSSA, owcor, high.rank.rate. Cả bốn phương pháp đều có bản số phức đầy đủ cho CSSA.
- Tương quan và gom nhóm: wcor có dấu, wnorm, contributions, frobenius.cor, grouping.auto theo wcor và periodogram.
- Ước lượng: parestimate ESPRIT (LS/TLS, cột/hàng, ghép cặp diag/memp, nD), pairs, roots.
- Dự báo: lrr, rforecast/vforecast (reverse, base original, hai hướng MSSA, PSSA), bforecast, forecast/predict có khoảng tin cậy và khoảng dự báo, residuals.
- Khác: hmatr; nhân và lấy hàng/cột của ma trận Hankel, Hankel-block-Hankel, Toeplitz mà không dựng ma trận (hmatmul, hrows, hcols và các bản hbh, t); ssa.capabilities.
- Biểu đồ: values, vectors, paired, series, reconstruction, wcor, roots, ảnh vector 2D, hmatr, periodogram grouping, dendrogram, clplot, biểu đồ khoảng trống.

Các hàm còn lại của Rssa là `clone`, `precache`, `cleanup`. Chúng chỉ quản lý bộ nhớ đệm của đối tượng R, nên bên Rust không cần.

**Những chỗ Rssa bị lỗi, ssa-rs đã làm đúng:**
- `wcor` cho CSSA báo lỗi khi so sánh số phức. ssa-rs trả về mô-đun của tương quan phức.
- W-OSSA kết hợp PSSA bị đệ quy vô hạn. ssa-rs tính các thành phần đặc biệt như PSSA, rồi phân rã phần còn lại với trọng số xiên. Cộng tất cả thành phần lại được đúng chuỗi gốc.
- I-OSSA, F-OSSA, E-OSSA, owcor trên CSSA bỏ mất phần ảo. ssa-rs tính toàn bộ bằng số phức. Với hai sóng phức có tần số sát nhau, E-OSSA tách đúng tới 1e-6, còn I-OSSA giảm sai số tách hơn một nửa.
- CSSA trên chuỗi có NA không phân rã được (cả `svd` và `eigen` đều báo lỗi), nên `gapfill`/`igapfill` cho CSSA trong Rssa không dùng được. ssa-rs phân rã trên các vector trễ đầy đủ. Trên chuỗi thực, kết quả khớp `gapfill` và `igapfill` của Rssa tới 1e-7. Trên tín hiệu phức chính xác, các khoảng trống được khôi phục tới 1e-8.
- `igapfill` cho MSSA nhập dạng list bị lỗi chỉ số. ssa-rs xử lý được cả trường hợp này.

## 5. Bản Python

**Đã sửa trong py-ssa-lib** (`python/py_ssa_lib/`, có test hồi quy trong `test/test_fixes.py`):
- `L_forecast`/`K_forecast` với `forecast_steps=0` trả về cả chuỗi thay vì mảng rỗng (`y_pred[-0:]`); số bước âm báo lỗi khó hiểu.
- Dữ liệu có NaN/inf làm SVD báo "did not converge" mà không nói nguyên nhân.
- Trọng số sai độ dài, bằng 0 hoặc âm: lúc báo `IndexError`, lúc cho ra NaN lặng lẽ.
- `L` không nguyên (ví dụ 30.5) báo `TypeError` ở chỗ xa nguyên nhân.
- Ma trận W-correlation ra NaN khi có thành phần chuẩn bằng 0 (chuỗi hằng).
- `estimate_LRR` chia cho 0 khi không gian con chứa vector đơn vị cuối; chuỗi toàn 0 gây chia cho 0 ở phần đóng góp.
- Chỉ số thành phần: giữ kiểu numpy (số âm, mặt nạ bool) nhưng báo `IndexError` rõ ràng khi sai.
- `bootstrap_prediction_intervals`: không đặt được seed, không kiểm tra `alpha`, `N_sims`, phần dư rỗng/NaN, dự báo 2 chiều bị lỗi khó hiểu. Thêm tham số `random_state`.
- `DistanceWeights`, `ClusterAwareWeights`: kiểm tra `bandwidth`, `cluster_penalty`, độ dài nhãn.

**Gói `ssa_rs` (Python + lõi Rust)** nay có đủ các tính năng Rssa của bản Rust, qua `ssa_rs.rssa`:
`NdSsa` (1D/2D/nD, cửa sổ hình tuỳ ý, NA, topo vòng, Toeplitz, Projection SSA, W-OSSA, ESPRIT diag/memp, I/F/E-OSSA, Cadzow, igapfill), `Cssa` (kể cả NA, gapfill, OSSA số phức), `MssaList`, `ShapedSsa`, `hmatr`, `summarize_gaps`. Các lớp `SSA`/`MSSA` thêm `wcor`, `forecast` (có khoảng tin cậy), `bforecast`, `rforecast`/`vforecast`, `parestimate`, `cadzow`, `fossa`/`iossa`/`eossa`/`owcor`, `grouping_auto_wcor`/`grouping_auto_pgram`. 13 test Python đối chiếu trực tiếp với số liệu do Rssa tính.

## 6. Chạy lại

```
cargo test --release --manifest-path rust/Cargo.toml
python -m pytest python/ssa_rs/tests python/py_ssa_lib/test   # 68 test
python python/bench/bench.py
cargo run --release --example bench_rssa --manifest-path rust/Cargo.toml -- rust/benchmarks/rust_times.json
python rust/benchmarks/make_tables.py
cargo run --release --example bench_rssa_ext --manifest-path rust/Cargo.toml -- rust/benchmarks/rust_times_ext.json
python rust/benchmarks/make_tables.py _ext
```
