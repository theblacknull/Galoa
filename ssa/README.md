# ssa-rs: py-ssa-lib viết lại bằng Rust

Bản viết lại bằng Rust của thư viện Python **py-ssa-lib 2.0.2** (SSA/MSSA: phân tích, tái tạo, dự báo và ước lượng nghiệm tín hiệu theo phương pháp của Golyandina – Nekrutkin – Zhigljavsky).

```
ssa/
├── rust/                      bản Rust (thuần Rust, chạy độc lập, không cần Python hay R)
│   ├── src/                       ssa, svd, hmat, lanczos, ndfft, vis ... và src/rssa/ (các tính năng của Rssa)
│   ├── tests/                     test Rust + tests/data (số liệu tham chiếu do Rssa tính)
│   ├── examples/                  chương trình đo tốc độ
│   ├── benchmarks/                thời gian đo được (Rssa và ssa-rs) + bảng so sánh
│   └── datasets/
├── python/                    bản Python
│   ├── py_ssa_lib/                thư viện py-ssa-lib 2.0.2 thuần Python (đã sửa lỗi) + test
│   ├── ssa_rs/                    gói Python gọi lõi Rust: thay thế py_ssa_lib và thêm ssa_rs.rssa
│   │   ├── src/                       binding PyO3 (biên dịch thành ssa_rs._core, dùng ../../rust)
│   │   ├── ssa_rs/                    mã Python của gói
│   │   └── tests/                     test đối chiếu py_ssa_lib và Rssa
│   └── bench/                     benchmark Python so với Rust
├── README.md
└── BAO_CAO_CAI_THIEN.md       báo cáo khả năng cải thiện
```

## Dùng trong Rust

```rust
use ssa_rs::{Ssa, Mssa, FullSvd, PrecomputedWeights};

let model = Ssa::new().fit(&series, Some(120))?;           // L = 120 (None = floor((N+1)/2))
let trend = model.reconstruct_components(&[0])?;           // (N, 1)
let w = model.compute_weighted_correlation_matrix();       // (d, d)
let fc = model.l_forecast(24, &[0, 1, 2], false)?;         // (24, 1)
let (mu, rho, omega) = model.estimate_esprit(&[1, 2])?;
```

## Dùng trong Python (thay thế trực tiếp py-ssa-lib)

```bash
pip install maturin
maturin develop --release -m python/ssa_rs/Cargo.toml
```

```python
# trước: from py_ssa_lib.ssa_collection import SSA, MSSA
from ssa_rs.ssa_collection import SSA, MSSA
from ssa_rs.svd_strategies import FullSVD, RandomizedSVD, EigenSVD
from ssa_rs.weights_strategies import PrecomputedWeights
from ssa_rs import vis_tools, stat_tools
```

## Kiểm thử

```bash
cargo test --release --manifest-path rust/Cargo.toml
```

```bash
python -m pytest python/ssa_rs/tests python/py_ssa_lib/test
```

## Benchmark

```bash
python python/bench/bench.py
```

Giấy phép: GPL-3.0-or-later, giống thư viện gốc, vì đây là bản phái sinh.
