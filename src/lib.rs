extern crate wgpu_array as ga;
extern crate ndarray;

pub use graph::{Graph, NodeIndex};
pub use var_store::VarIndex;
pub use op::Operation;
//pub use train::Trainer;

pub mod graph;
pub mod init;
//pub mod layers;
pub mod op;
//pub mod train;
//pub mod util;
pub mod var_store;

pub type CpuArray<T> = ndarray::Array<T, ndarray::IxDyn>;
