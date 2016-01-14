use std::cell::{Ref, RefCell, RefMut};

use matrix::ClMatrix;

use super::graph::Graph;

pub struct VarStore {
    vars: Vec<RefCell<ClMatrix<f32>>>,
}

impl VarStore {
    pub fn new() -> Self {
        VarStore {
            vars: vec![],
        }
    }

    pub fn add(&mut self, v: ClMatrix<f32>) -> VarIndex {
        self.vars.push(RefCell::new(v));
        VarIndex(self.vars.len()-1)
    }

    pub fn get<'a>(&'a self, v: VarIndex) -> Ref<'a, ClMatrix<f32>> {
        self.vars[v.0].borrow()
    }

    pub fn get_mut<'a>(&'a self, v: VarIndex) -> RefMut<'a, ClMatrix<f32>> {
        self.vars[v.0].borrow_mut()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct VarIndex(usize);

impl VarIndex {
    pub fn get<'a>(&self, g: &'a Graph) -> Ref<'a, ClMatrix<f32>> {
        g.var_store.get(*self)
    }

    pub fn get_mut<'a>(&self, g: &'a mut Graph) -> RefMut<'a, ClMatrix<f32>> {
        g.var_store.get_mut(*self)
    }
}
