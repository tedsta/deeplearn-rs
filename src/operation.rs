use matrix::{self, ClMatrix};
use matrix::cl_matrix::ClMatrixMode;

use super::graph::{Node, VarIndex, VarStore};

pub trait Operation {
    fn forward(&mut self, &matrix::Context, &mut VarStore, &mut Node);
    fn backward(&mut self, &matrix::Context, &mut VarStore, &mut Node, VarIndex);
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
        let event = {
            let a = v.get(n.inputs[0]);
            let b = v.get(n.inputs[1]);
            let c = v.get(n.outputs[0]);
            a.cross(ctx, b, c) // c = a*b
        };
        n.out_events.push(event);
    }

    fn backward(&mut self, ctx: &matrix::Context, v: &mut VarStore, n: &mut Node, grad: VarIndex) {
        // Derivative with respect to first input
        let (a_event, b_event) = {
            let a = v.get(n.inputs[0]);
            let b = v.get(n.inputs[1]);
            let a_d = v.get(n.gradients[0]);
            let b_d = v.get(n.gradients[1]);
            let g = v.get(grad);

            // a_d = g*b_t
            b.transpose(ctx, &self.b_t);
            let a_event = g.cross(ctx, &self.b_t, a_d);

            // b_d = a_t*g
            a.transpose(ctx, &self.a_t);
            let b_event = self.a_t.cross(ctx, g, b_d);

            (a_event, b_event)
        };
        n.out_events.push(a_event);
        n.out_events.push(b_event);
    }
}
