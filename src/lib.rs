extern crate matrix;

pub use graph::{Graph, NodeIndex};
pub use var_store::VarIndex;
pub use op::Operation;

pub mod graph;
pub mod op;
pub mod var_store;
