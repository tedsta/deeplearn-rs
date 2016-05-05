#![feature(zero_one)]

#[macro_use] extern crate gpuarray as ga;
extern crate rand;

pub use graph::{Graph, NodeIndex};
pub use var_store::VarIndex;
pub use op::Operation;
pub use trainer::Trainer;

pub mod graph;
pub mod init;
pub mod layers;
pub mod op;
pub mod trainer;
pub mod util;
pub mod var_store;
