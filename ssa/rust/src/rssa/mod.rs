//! Methods of the R package **Rssa** (Golyandina, Korobeynikov, Shlemov,
//! Usevich) that py-ssa-lib does not have, ported to Rust.
//!
//! | Rssa                                   | here                                   |
//! |----------------------------------------|----------------------------------------|
//! | `svd.method = "nutrlan"/"propack"`     | [`crate::svd::LanczosSvd`] + [`crate::hmat`] |
//! | `kind = "toeplitz-ssa"`                | [`crate::svd::ToeplitzSvd`]            |
//! | `kind = "cssa"`                        | [`cssa::Cssa`]                         |
//! | CSSA `gapfill`, `igapfill`, `iossa`, `fossa`, `eossa`, `owcor`, `wnorm` | [`cssa_ext`] |
//! | `kind = "2d-ssa"`                      | [`ssa2d::Ssa2d`]                       |
//! | `column/row.projector` (PSSA)          | [`pssa::Pssa`]                         |
//! | `column/row.oblique` (W-OSSA)          | [`pssa::Wossa`], with projectors [`pssa::wossa_pssa`] |
//! | `ssa.capabilities`                     | [`misc::capabilities`]                 |
//! | `hmatmul`, `hrows`, `hcols` (+ hbh, t) | [`crate::hmat::op_mul`], [`crate::hmat::op_row`], [`crate::hmat::op_col`] |
//! | shaped SSA with `NA`, `gapfill`        | [`gaps::ShapedSsa`]                    |
//! | `igapfill`, `summarize.gaps`, `clplot` | [`gaps`]                               |
//! | `cadzow`                               | `SsaModel::cadzow`                     |
//! | `iossa`, `fossa`, `eossa`, `owcor`     | [`ossa`]                               |
//! | `wcor` (signed), `frobenius.cor`, `wnorm`, `contributions` | [`grouping`]      |
//! | `grouping.auto.wcor/pgram`             | [`grouping`]                           |
//! | `parestimate` (ESPRIT LS/TLS, pairs)   | [`parest`]                             |
//! | `lrr`, `roots`, `rforecast`, `vforecast`, `bforecast` | [`forecast`]            |

pub mod common;
pub mod cssa;
pub mod cssa_ext;
pub mod forecast;
pub mod gaps;
pub mod grouping;
pub mod nd;
pub mod nd_ossa;
pub mod mssa_list;
pub mod misc;
pub mod ossa;
pub mod parest;
pub mod pssa;
pub mod ssa2d;

pub use common::Group;
pub use forecast::{apply_lrr, lrr_from_basis, lrr_roots, Base, Direction};
pub use parest::{roots2pars, Params, SolveMethod, Subspace};
