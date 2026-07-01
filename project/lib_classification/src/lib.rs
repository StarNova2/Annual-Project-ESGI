extern crate core;

mod linear;
mod linear_dataset;
mod mlp;
mod mlp_ffi;
mod rbf;
mod rbf_dataset;

// mod svm;
// mod svm_ffi;

pub use linear::*;
pub use linear_dataset::*;
pub use mlp_ffi::*;
pub use rbf::*;
// pub use svm_ffi::*;
