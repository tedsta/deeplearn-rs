use ga::{self, Tensor};
use ga::tensor::TensorMode;

use super::graph::Node;
use super::var_store::{VarIndex, VarStore};

pub trait Operation : 'static {
    fn forward(&mut self, &ga::Context, &mut VarStore, &mut Node);
    fn backward(&mut self, &ga::Context, &mut VarStore, &mut Node);
}

pub trait OpBuilder {
    type Op;

    fn build(&self, ctx: &ga::Context, v: &VarStore)
             -> Result<(Self::Op, Vec<VarIndex>, Vec<Vec<usize>>), String>;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct MatMul(pub VarIndex, pub VarIndex);

impl OpBuilder for MatMul {
    type Op = MatMulImpl;

    fn build(&self, ctx: &ga::Context, v: &VarStore)
             -> Result<(MatMulImpl, Vec<VarIndex>, Vec<Vec<usize>>), String> {
        let a = &v.get(self.0);
        let b = &v.get(self.1);
        Ok((MatMulImpl::new(ctx, a.shape().to_vec(), b.shape().to_vec()),
            vec![self.0, self.1],
            vec![vec![a.shape()[0], b.shape()[1]]]))
    }
}

pub struct MatMulImpl {
    a_t: Tensor<f32>,
    b_t: Tensor<f32>,
}

impl MatMulImpl {
    pub fn new(ctx: &ga::Context, a_shape: Vec<usize>, b_shape: Vec<usize>) -> Self {
        MatMulImpl {
            a_t: Tensor::new(ctx, vec![a_shape[1], a_shape[0]], TensorMode::Mut),
            b_t: Tensor::new(ctx, vec![b_shape[1], b_shape[0]], TensorMode::Mut),
        }
    }
}

impl Operation for MatMulImpl {
    fn forward(&mut self, ctx: &ga::Context, v: &mut VarStore, n: &mut Node) {
        let a = &v.get(n.inputs[0]);
        let b = &v.get(n.inputs[1]);
        let c = &v.get(n.outputs[0]);
        a.dot(ctx, b, c); // c = a*b
    }

    fn backward(&mut self, ctx: &ga::Context, v: &mut VarStore, n: &mut Node) {
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

pub struct Add(pub VarIndex, pub VarIndex, pub i32);

impl OpBuilder for Add {
    type Op = AddImpl;

    fn build(&self, _: &ga::Context, v: &VarStore)
             -> Result<(AddImpl, Vec<VarIndex>, Vec<Vec<usize>>), String> {
        let a = &v.get(self.0);
        let b = &v.get(self.1);
        if a.shape() != b.shape() {
            return Err("DIM ERROR: Shapes must be equal for Add".to_string());
        }
        Ok((AddImpl::new(self.2), vec![self.0, self.1], vec![a.shape().to_vec()]))
    }
}

pub struct AddImpl {
    axis: i32,
}

impl AddImpl {
    pub fn new(axis: i32) -> Self {
        AddImpl {
            axis: axis,
        }
    }
}

impl Operation for AddImpl {
    fn forward(&mut self, ctx: &ga::Context, v: &mut VarStore, n: &mut Node) {
        let a = &v.get(n.inputs[0]);
        let b = &v.get(n.inputs[1]);
        let c = &v.get(n.outputs[0]);
        a.add(ctx, self.axis, b, c); // c = a+b
    }

    fn backward(&mut self, ctx: &ga::Context, v: &mut VarStore, n: &mut Node) {
        let a_d = &v.get(n.in_grad[0]);
        let b_d = &v.get(n.in_grad[1]);
        let g = &v.get(n.out_grad[0].gradient());
        g.copy_to(ctx, a_d);
        g.sum(ctx, self.axis as usize, b_d);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct Relu(pub VarIndex);

impl OpBuilder for Relu {
    type Op = ReluImpl;

    fn build(&self, _: &ga::Context, v: &VarStore)
             -> Result<(ReluImpl, Vec<VarIndex>, Vec<Vec<usize>>), String> {
        let a = &v.get(self.0);
        Ok((ReluImpl, vec![self.0], vec![a.shape().to_vec()]))
    }
}

pub struct ReluImpl;

impl Operation for ReluImpl {
    fn forward(&mut self, ctx: &ga::Context, v: &mut VarStore, n: &mut Node) {
        let a = &v.get(n.inputs[0]);
        let b = &v.get(n.outputs[0]);
        a.max(ctx, 0.0, b); // b = max(0, a)
    }

    fn backward(&mut self, ctx: &ga::Context, v: &mut VarStore, n: &mut Node) {
        let a = &v.get(n.inputs[0]);
        let a_d = &v.get(n.in_grad[0]);
        let g = &v.get(n.out_grad[0].gradient());
        a.dmax(ctx, 0.0, a_d);
        g.multiply(ctx, -1, a_d, a_d);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct Mse(pub VarIndex, pub VarIndex);

impl OpBuilder for Mse {
    type Op = MseImpl;

    fn build(&self, _: &ga::Context, v: &VarStore)
             -> Result<(MseImpl, Vec<VarIndex>, Vec<Vec<usize>>), String> {
        let a = &v.get(self.0);
        let b = &v.get(self.1);
        if a.shape() != b.shape() {
            return Err("DIM ERROR: Shapes must be equal for MSE".to_string());
        }
        Ok((MseImpl, vec![self.0, self.1], vec![a.shape().to_vec()]))
    }
}

pub struct MseImpl;

impl Operation for MseImpl {
    fn forward(&mut self, ctx: &ga::Context, v: &mut VarStore, n: &mut Node) {
        let h = &v.get(n.inputs[0]); // predictions
        let y = &v.get(n.inputs[1]); // training output
        let out = &v.get(n.outputs[0]);
        h.mse(ctx, y, out); // out = mse(h, y)
    }

    fn backward(&mut self, ctx: &ga::Context, v: &mut VarStore, n: &mut Node) {
        let h = &v.get(n.inputs[0]); // predictions
        let h_d = &v.get(n.in_grad[0]);
        let y = &v.get(n.inputs[1]); // training output
        let g = &v.get(n.out_grad[0].gradient());
        h.dmse(ctx, y, h_d); // h_d = dmse(h, y)
        h_d.multiply(ctx, 0, g, h_d); // h_d = g*h_d
    }
}
