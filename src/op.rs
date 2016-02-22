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
        ga::matmul(ctx, a, b, c); // c = a*b
    }

    fn backward(&mut self, ctx: &ga::Context, v: &mut VarStore, n: &mut Node) {
        let a = &v.get(n.inputs[0]);
        let b = &v.get(n.inputs[1]);
        let a_d = &v.get(n.in_grad[0]);
        let b_d = &v.get(n.in_grad[1]);
        let g = &v.get(n.out_grad[0].gradient());
        
        // Derivative with respect to first input
        // a_d = g*b_t
        ga::transpose(ctx, b, &self.b_t);
        ga::matmul(ctx, g, &self.b_t, a_d);

        // Derivative with respect to second input
        // b_d = a_t*g
        ga::transpose(ctx, a, &self.a_t);
        ga::matmul(ctx, &self.a_t, g, b_d);
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
        ga::add(ctx, a, self.axis, b, c); // c = a+b
    }

    fn backward(&mut self, ctx: &ga::Context, v: &mut VarStore, n: &mut Node) {
        let a_d = &v.get(n.in_grad[0]);
        let b_d = &v.get(n.in_grad[1]);
        let g = &v.get(n.out_grad[0].gradient());
        ga::copy_to(ctx, g, a_d);
        ga::sum(ctx, g, self.axis as usize, b_d);
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
        ga::max(ctx, a, 0.0, b); // b = max(0, a)
    }

    fn backward(&mut self, ctx: &ga::Context, v: &mut VarStore, n: &mut Node) {
        let a = &v.get(n.inputs[0]);
        let a_d = &v.get(n.in_grad[0]);
        let g = &v.get(n.out_grad[0].gradient());
        ga::dmax(ctx, a, 0.0, a_d);
        ga::multiply(ctx, g, -1, a_d, a_d);
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
        ga::mse(ctx, h, y, out); // out = mse(h, y)
    }

    fn backward(&mut self, ctx: &ga::Context, v: &mut VarStore, n: &mut Node) {
        let h = &v.get(n.inputs[0]); // predictions
        let h_d = &v.get(n.in_grad[0]);
        let y = &v.get(n.inputs[1]); // training output
        let g = &v.get(n.out_grad[0].gradient());
        ga::dmse(ctx, h, y, h_d); // h_d = dmse(h, y)
        ga::multiply(ctx, h_d, 0, g, h_d); // h_d = g*h_d
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Lstm(input, state)
/*pub struct Lstm(pub VarIndex, pub VarIndex, usize);

impl OpBuilder for Lstm {
    type Op = LstmImpl;

    fn build(&self, _: &ga::Context, v: &VarStore)
             -> Result<(LstmImpl, Vec<VarIndex>, Vec<Vec<usize>>), String> {
        let x = &v.get(self.0);
        let s = &v.get(self.1);
        let hidden_size = self.2;
        if x.shape() != s.shape() {
            return Err("DIM ERROR: Shapes must be equal for LSTM".to_string());
        }
        Ok((LstmImpl::new(), vec![self.0, self.1], vec![x.shape().to_vec()]))
    }
}

pub struct LstmImpl {
    ifog: Tensor<f32>, // input, forget, output, gate (IFOG)
    
    // Recurrent connections from last sequence
    c0: Tensor<f32>,
    h0: Tensor<f32>,

    seq_len: usize,
    batch_size: usize,
    input_size: usize,
    hidden_size: usize,
}

impl LstmImpl {
    fn new(seq_len: usize, batch_size: usize, input_size: usize, hidden_size: usize) -> Self {
        let n = seq_len;
        let b = batch_size;
        let d = hidden_size;
        LstmImpl {
            ifog: Tensor::new(ctx, vec![n, b, d*4], TensorMode::Mut)),
            c0: Tensor::new(ctx, vec![b, d], TensorMode::Mut)),
            h0: Tensor::new(ctx, vec![b, d], TensorMode::Mut)),

            seq_len: seq_len,
            batch_size: batch_size,
            input_size: input_size,
            hidden_size: hidden_size,
        }
    }
}

impl Operation for LstmImpl {
    fn forward(&mut self, ctx: &ga::Context, v: &mut VarStore, n: &mut Node) {
        let x = &v.get(n.inputs[0]); // input
        let c = &v.get(n.outputs[0]); // cell
        let h = &v.get(n.outputs[1]); // output

        for t in 0..self.seq_len {
            self.ifog[t].dot(ctx, h);
            h.mse(ctx, y, out); // out = mse(h, y)
        }
    }

    fn backward(&mut self, ctx: &ga::Context, v: &mut VarStore, n: &mut Node) {
    }
}*/
