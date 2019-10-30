use futures_channel::oneshot;
use ga;

use crate::CpuArray;
use super::graph::Graph;

pub struct VarStore {
    vars: Vec<ga::Array<f32>>,
}

impl VarStore {
    pub fn new() -> Self {
        VarStore {
            vars: vec![],
        }
    }

    pub fn add(&mut self, v: ga::Array<f32>) -> VarIndex {
        self.vars.push(v);
        VarIndex(self.vars.len() - 1)
    }

    pub fn get<'a>(&'a self, v: VarIndex) -> &ga::Array<f32> {
        &self.vars[v.0]
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct VarIndex(usize);

impl VarIndex {
    pub fn get<'a>(self, g: &'a Graph) -> &'a ga::Array<f32> {
        g.var_store.get(self)
    }

    pub fn read(self, g: &Graph, a: CpuArray<f32>) -> oneshot::Receiver<CpuArray<f32>> {
        g.var_store.get(self).read(a)
    }

    pub fn write(self, g: &Graph, a: CpuArray<f32>) -> oneshot::Receiver<CpuArray<f32>> {
        g.var_store.get(self).write(a)
    }
}
