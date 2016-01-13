use matrix::{self, ClMatrix};
use matrix::cl_matrix::ClMatrixMode;

use super::graph::{Graph, NodeIndex, VarIndex};

pub trait Operation {
    fn forward(&mut self, &matrix::Context, &mut Graph, NodeIndex);
    fn backward(&mut self, &matrix::Context, &mut Graph, NodeIndex, VarIndex);
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
    fn forward(&mut self, ctx: &matrix::Context, g: &mut Graph, n: NodeIndex) {
        let event = {
            let node = n.get(g);
            let a = node.inputs[0].get(g);
            let b = node.inputs[1].get(g);
            let c = node.outputs[0].get(g);
            a.cross(ctx, b, c) // c = a*b
        };
        n.get_mut(g).out_events.push(event);
    }

    fn backward(&mut self, ctx: &matrix::Context, g: &mut Graph, n: NodeIndex, grad: VarIndex) {
        // Derivative with respect to first input
        let (a_event, b_event) = {
            let node = n.get(g);
            let a = node.inputs[0].get(g);
            let b = node.inputs[1].get(g);
            let a_d = node.gradients[0].get(g);
            let b_d = node.gradients[1].get(g);
            let g = grad.get(g);

            // a_d = g*b_t
            b.transpose(ctx, &self.b_t);
            let a_event = g.cross(ctx, &self.b_t, a_d);

            // b_d = a_t*g
            a.transpose(ctx, &self.a_t);
            let b_event = self.a_t.cross(ctx, g, b_d);

            (a_event, b_event)
        };
        n.get_mut(g).out_events.push(a_event);
        n.get_mut(g).out_events.push(b_event);
    }
}
