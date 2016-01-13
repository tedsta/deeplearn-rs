use matrix::ClMatrix;

use super::graph::Graph;

pub struct VarStore {
    vars: Vec<ClMatrix<f32>>,
}

impl VarStore {
    pub fn new() -> Self {
        VarStore { vars: vec![] }
    }

    pub fn add(&mut self, v: ClMatrix<f32>) -> VarIndex {
        self.vars.push(v);
        VarIndex(self.vars.len()-1)
    }

    pub fn get<'a>(&'a self, v: VarIndex) -> &'a ClMatrix<f32> {
        &self.vars[v.0]
    }

    pub fn get_mut<'a>(&'a mut self, v: VarIndex) -> &'a mut ClMatrix<f32> {
        &mut self.vars[v.0]
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone)]
pub struct VarIndex(usize);

impl VarIndex {
    pub fn get<'a>(&self, g: &'a Graph) -> &'a ClMatrix<f32> {
        g.var_store.get(*self)
    }

    pub fn get_mut<'a>(&self, g: &'a mut Graph) -> &'a mut ClMatrix<f32> {
        g.var_store.get_mut(*self)
    }
}
