use matrix::{self, ClMatrix};
use matrix::cl_matrix::ClMatrixMode;

use super::graph::Node;
use super::var_store::VarStore;

pub trait Operation {
    fn forward(&mut self, &matrix::Context, &mut VarStore, &mut Node);
    fn backward(&mut self, &matrix::Context, &mut VarStore, &mut Node);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct MatMul {
    a_t: ClMatrix<f32>,
    b_t: ClMatrix<f32>,
}

impl MatMul {
    pub fn new(ctx: &matrix::Context, a_shape: (u64, u64), b_shape: (u64, u64)) -> Self {
        MatMul {
            a_t: ClMatrix::new(ctx, a_shape.1 as usize, a_shape.0 as usize, ClMatrixMode::Mut),
            b_t: ClMatrix::new(ctx, b_shape.1 as usize, b_shape.0 as usize, ClMatrixMode::Mut),
        }
    }
}

impl Operation for MatMul {
    fn forward(&mut self, ctx: &matrix::Context, v: &mut VarStore, n: &mut Node) {
        let a = &v.get(n.inputs[0]);
        let b = &v.get(n.inputs[1]);
        let c = &mut v.get_mut(n.outputs[0]);
        a.cross(ctx, b, c); // c = a*b
    }

    fn backward(&mut self, ctx: &matrix::Context, v: &mut VarStore, n: &mut Node) {
        let a = &v.get(n.inputs[0]);
        let b = &v.get(n.inputs[1]);
        let a_d = &mut v.get_mut(n.in_grad[0]);
        let b_d = &mut v.get_mut(n.in_grad[1]);
        let g = &v.get(n.out_grad[0].gradient());
        
        // Derivative with respect to first input
        // a_d = g*b_t
        b.transpose(ctx, &mut self.b_t);
        g.cross(ctx, &self.b_t, a_d);

        // Derivative with respect to second input
        // b_d = a_t*g
        a.transpose(ctx, &mut self.a_t);
        self.a_t.cross(ctx, g, b_d);
    }
}
