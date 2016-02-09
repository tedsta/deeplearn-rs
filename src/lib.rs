extern crate matrix;
extern crate rand;

pub use graph::{Graph, NodeIndex};
pub use var_store::VarIndex;
pub use op::Operation;

pub mod graph;
pub mod init;
pub mod op;
pub mod var_store;
