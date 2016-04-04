use std::cell::{Ref, RefCell, RefMut};

use ga::{Array, Tensor};

use super::graph::Graph;

pub struct VarStore {
    vars: Vec<RefCell<Tensor<f32>>>,
}

impl VarStore {
    pub fn new() -> Self {
        VarStore {
            vars: vec![],
        }
    }

    pub fn add(&mut self, v: Tensor<f32>) -> VarIndex {
        self.vars.push(RefCell::new(v));
        VarIndex(self.vars.len()-1)
    }

    pub fn get<'a>(&'a self, v: VarIndex) -> Ref<'a, Tensor<f32>> {
        self.vars[v.0].borrow()
    }

    pub fn get_mut<'a>(&'a self, v: VarIndex) -> RefMut<'a, Tensor<f32>> {
        self.vars[v.0].borrow_mut()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct VarIndex(usize);

impl VarIndex {
    pub fn get<'a>(self, g: &'a Graph) -> Ref<'a, Tensor<f32>> {
        g.var_store.get(self)
    }

    pub fn get_cpu(self, g: &Graph) -> Array<f32> {
        g.var_store.get(self).get(g.context())
    }

    pub fn get_mut<'a>(self, g: &'a mut Graph) -> RefMut<'a, Tensor<f32>> {
        g.var_store.get_mut(self)
    }
}
