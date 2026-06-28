extern crate core;

mod linear;
mod linear_dataset;
mod mlp;
mod mlp_ffi;
mod RBF;

mod RBF_database;
// mod svm;
// mod svm_ffi;

pub use linear::*;
pub use linear_dataset::*;
pub use mlp_ffi::*;
pub use RBF::*;
// pub use svm_ffi::*;
