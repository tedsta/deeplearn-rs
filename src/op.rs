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
        let c = &v.get(n.outputs[0]);
        a.dot(ctx, b, c); // c = a*b
    }

    fn backward(&mut self, ctx: &matrix::Context, v: &mut VarStore, n: &mut Node) {
        let a = &v.get(n.inputs[0]);
        let b = &v.get(n.inputs[1]);
        let a_d = &v.get(n.in_grad[0]);
        let b_d = &v.get(n.in_grad[1]);
        let g = &v.get(n.out_grad[0].gradient());
        
        // Derivative with respect to first input
        // a_d = g*b_t
        b.transpose(ctx, &self.b_t);
        g.dot(ctx, &self.b_t, a_d);

        // Derivative with respect to second input
        // b_d = a_t*g
        a.transpose(ctx, &self.a_t);
        self.a_t.dot(ctx, g, b_d);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct Add {
    axis: i32,
}

impl Add {
    pub fn new(axis: i32) -> Self {
        Add {
            axis: axis,
        }
    }
}

impl Operation for Add {
    fn forward(&mut self, ctx: &matrix::Context, v: &mut VarStore, n: &mut Node) {
        let a = &v.get(n.inputs[0]);
        let b = &v.get(n.inputs[1]);
        let c = &v.get(n.outputs[0]);
        a.add(ctx, self.axis, b, c); // c = a+b
    }

    fn backward(&mut self, ctx: &matrix::Context, v: &mut VarStore, n: &mut Node) {
        let a_d = &v.get(n.in_grad[0]);
        let b_d = &v.get(n.in_grad[1]);
        let g = &v.get(n.out_grad[0].gradient());
        g.copy_to(ctx, a_d);
        g.copy_to(ctx, b_d);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct Relu;

impl Relu {
    pub fn new() -> Self {
        Relu
    }
}

impl Operation for Relu {
    fn forward(&mut self, ctx: &matrix::Context, v: &mut VarStore, n: &mut Node) {
        let a = &v.get(n.inputs[0]);
        let b = &v.get(n.outputs[0]);
        a.max(ctx, 0.0, b); // b = max(0, a)
    }

    fn backward(&mut self, ctx: &matrix::Context, v: &mut VarStore, n: &mut Node) {
        let a = &v.get(n.inputs[0]);
        let a_d = &v.get(n.in_grad[0]);
        let g = &v.get(n.out_grad[0].gradient());
        a.dmax(ctx, 0.0, a_d);
        g.multiply(ctx, a_d, a_d);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct Mse;

impl Mse {
    pub fn new() -> Self {
        Mse
    }
}

impl Operation for Mse {
    fn forward(&mut self, ctx: &matrix::Context, v: &mut VarStore, n: &mut Node) {
        let h = &v.get(n.inputs[0]); // predictions
        let y = &v.get(n.inputs[1]); // training output
        let out = &v.get(n.outputs[0]);
        h.mse(ctx, y, out); // out = mse(h, y)
    }

    fn backward(&mut self, ctx: &matrix::Context, v: &mut VarStore, n: &mut Node) {
        let h = &v.get(n.inputs[0]); // predictions
        let h_d = &v.get(n.in_grad[0]);
        let y = &v.get(n.inputs[1]); // training output
        let g = &v.get(n.out_grad[0].gradient());
        h.dmse(ctx, y, h_d); // h_d = dmse(h, y)
        g.multiply(ctx, h_d, h_d); // h_d = g*h_d
    }
}
