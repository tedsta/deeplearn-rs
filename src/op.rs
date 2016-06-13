use ga::{self, Tensor};
use ga::tensor::TensorMode;

use super::graph::{Node, NodeInput};
use super::var_store::{VarIndex, VarStore};

pub trait Operation : 'static {
    fn forward(&mut self, &ga::Context, &VarStore, &Node);
    fn backward(&mut self, &ga::Context, &VarStore, &Node);
    fn reset_rnn(&mut self, _: &ga::Context, _: &VarStore, _: &Node) { }
}

pub trait OpBuilder {
    type Op: Operation;

    fn build(&self, ctx: &ga::Context, v: &mut VarStore)
             -> Result<OpDescriptor<Self::Op>, String>;
}

pub struct OpDescriptor<T: Operation> {
    pub op: T,
    pub inputs: Vec<NodeInput>,
    pub out_shapes: Vec<Vec<usize>>,
    pub back_dep: Vec<VarIndex>, // Dependencies for backward pass
}

////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct MatMul(pub VarIndex, pub VarIndex);

impl OpBuilder for MatMul {
    type Op = MatMulImpl;

    fn build(&self, ctx: &ga::Context, v: &mut VarStore)
             -> Result<OpDescriptor<MatMulImpl>, String> {
        let a_shape = v.get(self.0).shape().to_vec();
        let b_shape = v.get(self.1).shape().to_vec();
        if a_shape[1] != b_shape[0] {
            return Err(format!("DIM ERROR: Shapes must be of form [I, J] and [J, K]
                                (got {:?} and {:?}) for MatMul",
                               a_shape, b_shape));
        }
        let out_shape = vec![a_shape[0], b_shape[1]];
        Ok(OpDescriptor {
            op: MatMulImpl::new(ctx, v, a_shape, b_shape),
            inputs: vec![NodeInput::Var(self.0), NodeInput::Var(self.1)],
            out_shapes: vec![out_shape],
            back_dep: vec![self.0, self.1],
        })
    }
}

pub struct MatMulImpl {
    a_t: VarIndex,
    b_t: VarIndex,
}

impl MatMulImpl {
    pub fn new(ctx: &ga::Context, v: &mut VarStore, a_shape: Vec<usize>, b_shape: Vec<usize>) -> Self {
        MatMulImpl {
            a_t: v.add(Tensor::new(ctx, vec![a_shape[1], a_shape[0]], TensorMode::Mut)),
            b_t: v.add(Tensor::new(ctx, vec![b_shape[1], b_shape[0]], TensorMode::Mut)),
        }
    }
}

impl Operation for MatMulImpl {
    fn forward(&mut self, ctx: &ga::Context, v: &VarStore, n: &Node) {
        let a = &v.get(n.inputs[0]);
        let b = &v.get(n.inputs[1]);
        let c = &v.get(n.outputs[0]);
        ga::matmul(ctx, a, b, c); // c = a*b
    }

    fn backward(&mut self, ctx: &ga::Context, v: &VarStore, n: &Node) {
        let a = &v.get(n.inputs[0]);
        let b = &v.get(n.inputs[1]);
        let a_d = &v.get(n.in_grad[0]);
        let b_d = &v.get(n.in_grad[1]);
        let g = &v.get(n.out_grad[0].get());
        
        // Derivative with respect to first input
        // a_d = g*b_t
        ga::transpose(ctx, b, &v.get(self.b_t));
        ga::matmul(ctx, g, &v.get(self.b_t), a_d);

        // Derivative with respect to second input
        // b_d = a_t*g
        ga::transpose(ctx, a, &v.get(self.a_t));
        ga::matmul(ctx, &v.get(self.a_t), g, b_d);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct Add(pub VarIndex, pub VarIndex, pub i32);

impl OpBuilder for Add {
    type Op = AddImpl;

    fn build(&self, _: &ga::Context, v: &mut VarStore)
             -> Result<OpDescriptor<AddImpl>, String> {
        let a = &v.get(self.0);
        let b = &v.get(self.1);
        let add_axis = self.2;
        match add_axis {
            -1 => {
                if a.shape() != b.shape() {
                    return Err("DIM ERROR: Shapes must be equal for Add".to_string());
                }
            },
            0 => {
                if b.shape()[0] != 1 || a.shape()[1] != b.shape()[1] {
                    return Err(format!("DIM ERROR: Shapes must be [M, N] and [1, N]
                                        (got {:?} and {:?}) for Add with broadcast axis of 0",
                                       a.shape(), b.shape()));
                }
            },
            1 => {
                if b.shape()[1] != 1 || a.shape()[0] != b.shape()[0] {
                    return Err(format!("DIM ERROR: Shapes must be [M, N] and [M, 1]
                                        (got {:?} and {:?}) for Add with broadcast axis of 1",
                                       a.shape(), b.shape()));
                }
            },
            _ => { 
                return Err(format!("BROADCAST AXIS ERROR: Invalid broadcast axis {}", add_axis));
            }
        }
        Ok(OpDescriptor {
            op: AddImpl::new(add_axis),
            inputs: vec![NodeInput::Var(self.0), NodeInput::Var(self.1)],
            out_shapes: vec![a.shape().to_vec()],
            back_dep: vec![],
        })
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
    fn forward(&mut self, ctx: &ga::Context, v: &VarStore, n: &Node) {
        let a = &v.get(n.inputs[0]);
        let b = &v.get(n.inputs[1]);
        let c = &v.get(n.outputs[0]);
        ga::add(ctx, a, self.axis, b, c); // c = a+b
    }

    fn backward(&mut self, ctx: &ga::Context, v: &VarStore, n: &Node) {
        let a_d = &v.get(n.in_grad[0]);
        let b_d = &v.get(n.in_grad[1]);
        let g = &v.get(n.out_grad[0].get());
        ga::copy_to(ctx, g, a_d);
        ga::sum(ctx, g, self.axis as usize, b_d);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Softmax(input)
pub struct Softmax(pub VarIndex);

impl OpBuilder for Softmax {
    type Op = SoftmaxImpl;

    fn build(&self, ctx: &ga::Context, v: &mut VarStore)
             -> Result<OpDescriptor<SoftmaxImpl>, String> {
        let batches = v.get(self.0).shape()[0];
        let classes = v.get(self.0).shape()[1];
        Ok(OpDescriptor {
            op: SoftmaxImpl::new(ctx, v, batches),
            inputs: vec![NodeInput::Var(self.0)],
            out_shapes: vec![vec![batches, classes]],
            back_dep: vec![],
        })
    }
}

pub struct SoftmaxImpl {
    exp_sum: VarIndex,
}

impl SoftmaxImpl {
    pub fn new(ctx: &ga::Context, v: &mut VarStore, batches: usize) -> Self {
        SoftmaxImpl {
            exp_sum: v.add(Tensor::new(ctx, vec![batches, 1], TensorMode::Mut)),
        }
    }
}

impl Operation for SoftmaxImpl {
    fn forward(&mut self, ctx: &ga::Context, v: &VarStore, n: &Node) {
        let input = &v.get(n.inputs[0]);
        let prob = &v.get(n.outputs[0]);

        let exp_sum = &v.get(self.exp_sum);

        ga::exp(ctx, input, prob);
        ga::sum(ctx, prob, 1, exp_sum);
        //println!("{:?}", input.get(ctx));
        //println!("{:?}", exp_sum.get(ctx)[&[0, 0]]);
        if exp_sum.get(ctx)[&[0, 0]].is_nan() {
            panic!("NaN :(");
        }
        if exp_sum.get(ctx)[&[0, 0]] == 0.0 {
            panic!("Zero :(");
        }
        ga::divide(ctx, prob, 1, exp_sum, prob);
    }

    fn backward(&mut self, ctx: &ga::Context, v: &VarStore, n: &Node) {
        let input_d = &v.get(n.in_grad[0]);
        let prob_d = &v.get(n.out_grad[0].get());

        ga::copy_to(ctx, prob_d, input_d);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// CrossEntropy(prob, true_prob)
pub struct CrossEntropy(pub VarIndex, pub VarIndex);

impl OpBuilder for CrossEntropy {
    type Op = CrossEntropyImpl;

    fn build(&self, _: &ga::Context, v: &mut VarStore)
             -> Result<OpDescriptor<CrossEntropyImpl>, String> {
        let prob = &v.get(self.0);
        let true_prob = &v.get(self.1);
        if prob.shape() != true_prob.shape() {
            return Err("DIM ERROR: Shapes must be equal for CrossEntropy".to_string());
        }
        Ok(OpDescriptor {
            op: CrossEntropyImpl::new(),
            inputs: vec![NodeInput::Var(self.0), NodeInput::Var(self.1)],
            out_shapes: vec![prob.shape().to_vec()],
            back_dep: vec![],
        })
    }
}

pub struct CrossEntropyImpl;

impl CrossEntropyImpl {
    pub fn new() -> Self {
        CrossEntropyImpl
    }
}

impl Operation for CrossEntropyImpl {
    fn forward(&mut self, ctx: &ga::Context, v: &VarStore, n: &Node) {
        let prob = &v.get(n.inputs[0]);
        let true_prob = &v.get(n.inputs[1]);
        let loss = &v.get(n.outputs[0]);

        ga::log(ctx, prob, loss);
        ga::negate(ctx, loss, loss);
        ga::multiply(ctx, loss, -1, true_prob, loss);
    }

    fn backward(&mut self, ctx: &ga::Context, v: &VarStore, n: &Node) {
        let prob = &v.get(n.inputs[0]);
        let true_prob = &v.get(n.inputs[1]);

        let prob_d = &v.get(n.in_grad[0]);
        let loss_d = &v.get(n.out_grad[0].get());

        ga::sub(ctx, prob, true_prob, prob_d);
        ga::multiply(ctx, prob_d, -1, loss_d, prob_d);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct Relu(pub VarIndex);

impl OpBuilder for Relu {
    type Op = ReluImpl;

    fn build(&self, _: &ga::Context, v: &mut VarStore)
             -> Result<OpDescriptor<ReluImpl>, String> {
        let a = &v.get(self.0);
        Ok(OpDescriptor {
            op: ReluImpl,
            inputs: vec![NodeInput::Var(self.0)],
            out_shapes: vec![a.shape().to_vec()],
            back_dep: vec![self.0],
        })
    }
}

pub struct ReluImpl;

impl Operation for ReluImpl {
    fn forward(&mut self, ctx: &ga::Context, v: &VarStore, n: &Node) {
        let a = &v.get(n.inputs[0]);
        let b = &v.get(n.outputs[0]);
        ga::max(ctx, a, 0.0, b); // b = max(0, a)
    }

    fn backward(&mut self, ctx: &ga::Context, v: &VarStore, n: &Node) {
        let a = &v.get(n.inputs[0]);
        let a_d = &v.get(n.in_grad[0]);
        let g = &v.get(n.out_grad[0].get());
        ga::dmax(ctx, a, 0.0, a_d);
        ga::multiply(ctx, g, -1, a_d, a_d);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct Mse(pub VarIndex, pub VarIndex);

impl OpBuilder for Mse {
    type Op = MseImpl;

    fn build(&self, _: &ga::Context, v: &mut VarStore)
             -> Result<OpDescriptor<MseImpl>, String> {
        let a = &v.get(self.0);
        let b = &v.get(self.1);
        if a.shape() != b.shape() {
            return Err("DIM ERROR: Shapes must be equal for MSE".to_string());
        }
        Ok(OpDescriptor {
            op: MseImpl,
            inputs: vec![NodeInput::Var(self.0), NodeInput::Var(self.1)],
            out_shapes: vec![a.shape().to_vec()],
            back_dep: vec![self.0, self.1],
        })
    }
}

pub struct MseImpl;

impl Operation for MseImpl {
    fn forward(&mut self, ctx: &ga::Context, v: &VarStore, n: &Node) {
        let h = &v.get(n.inputs[0]); // predictions
        let y = &v.get(n.inputs[1]); // training output
        let out = &v.get(n.outputs[0]);
        ga::mse(ctx, h, y, out); // out = mse(h, y)
    }

    fn backward(&mut self, ctx: &ga::Context, v: &VarStore, n: &Node) {
        let h = &v.get(n.inputs[0]); // predictions
        let h_d = &v.get(n.in_grad[0]);
        let y = &v.get(n.inputs[1]); // training output
        let g = &v.get(n.out_grad[0].get());
        ga::dmse(ctx, h, y, h_d); // h_d = dmse(h, y)
        ga::multiply(ctx, h_d, 0, g, h_d); // h_d = g*h_d
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Lstm(input, state)
pub struct Lstm(pub VarIndex, pub VarIndex, pub usize);

impl OpBuilder for Lstm {
    type Op = LstmImpl;

    fn build(&self, ctx: &ga::Context, v: &mut VarStore)
             -> Result<OpDescriptor<LstmImpl>, String> {
        let batch_size = v.get(self.0).shape()[0];
        let input_size = v.get(self.0).shape()[1];
        let hidden_size = self.2;
        if v.get(self.1).shape()[0] != 1+input_size+hidden_size || v.get(self.1).shape()[1] != 4*hidden_size {
            return Err(format!("DIM ERROR: LSTM expects weight matrix shape of
                                [1+input_size+hidden_size, 4*hidden_size], got {:?}",
                               v.get(self.1).shape()));
        }
        let lstm_impl = LstmImpl::new(ctx, v, batch_size, input_size, hidden_size);
        let h_in = lstm_impl.h_in;
        let ifog_f = lstm_impl.ifog_f;
        Ok(OpDescriptor {
            op: lstm_impl,
            inputs: vec![NodeInput::Var(self.0), NodeInput::Var(self.1),
                         NodeInput::Recurrent(0), NodeInput::Recurrent(1)],
            out_shapes: vec![vec![batch_size, hidden_size], vec![batch_size, hidden_size]],
            back_dep: vec![h_in, ifog_f],
        })
    }
}

// LstmUnrolled(input, state, prev_h, prev_c)
pub struct LstmUnrolled(pub VarIndex, pub VarIndex, pub VarIndex, pub VarIndex);

impl OpBuilder for LstmUnrolled {
    type Op = LstmImpl;

    fn build(&self, ctx: &ga::Context, v: &mut VarStore)
             -> Result<OpDescriptor<LstmImpl>, String> {
        let batch_size = v.get(self.0).shape()[0];
        let input_size = v.get(self.0).shape()[1];
        let hidden_size = v.get(self.2).shape()[1];
        if v.get(self.1).shape()[0] != 1+input_size+hidden_size || v.get(self.1).shape()[1] != 4*hidden_size {
            return Err(format!("DIM ERROR: LSTM expects weight matrix shape of
                                [1+input_size+hidden_size, 4*hidden_size], got {:?}",
                               v.get(self.1).shape()));
        }
        let lstm_impl = LstmImpl::new(ctx, v, batch_size, input_size, hidden_size);
        let h_in = lstm_impl.h_in;
        let ifog_f = lstm_impl.ifog_f;
        Ok(OpDescriptor {
            op: lstm_impl,
            inputs: vec![NodeInput::Var(self.0), NodeInput::Var(self.1),
                         NodeInput::Var(self.2), NodeInput::Var(self.3)],
            out_shapes: vec![vec![batch_size, hidden_size], vec![batch_size, hidden_size]],
            back_dep: vec![h_in, ifog_f],
        })
    }
}

pub struct LstmImpl {
    h_in: VarIndex, // Concatonated input and previous output
    d_h_in: VarIndex,
    ifog: VarIndex, // input, forget, output, gate (IFOG): input sums
    d_ifog: VarIndex,
    ifog_f: VarIndex, // input, forget, output, gate: activations
    d_ifog_f: VarIndex,
    d_c_inner: VarIndex,
    c_f: VarIndex, // tanh of C
    d_c_f: VarIndex,

    h_in_t: VarIndex,
    wlstm_t: VarIndex,

    input_size: usize,
    hidden_size: usize,
}

impl LstmImpl {
    fn new(ctx: &ga::Context,
           v: &mut VarStore,
           batch_size: usize,
           input_size: usize,
           hidden_size: usize) -> Self {
        let b = batch_size;
        let d = hidden_size;

        let h_in = Tensor::new(ctx, vec![b, 1 + input_size + d], TensorMode::Mut);

        // Fill first column of h_in with 1's to be multiplied by the biases in the weights matrix
        ga::fill_slice(ctx, &h_in.slice(s![.., 0]), 1.0);

        LstmImpl {
            h_in: v.add(h_in),
            d_h_in: v.add(Tensor::new(ctx, vec![b, 1 + input_size + d], TensorMode::Mut)),
            ifog: v.add(Tensor::new(ctx, vec![b, d*4], TensorMode::Mut)),
            d_ifog: v.add(Tensor::new(ctx, vec![b, d*4], TensorMode::Mut)),
            ifog_f: v.add(Tensor::new(ctx, vec![b, d*4], TensorMode::Mut)),
            d_ifog_f: v.add(Tensor::new(ctx, vec![b, d*4], TensorMode::Mut)),
            d_c_inner: v.add(Tensor::new(ctx, vec![b, d], TensorMode::Mut)),
            c_f: v.add(Tensor::new(ctx, vec![b, d], TensorMode::Mut)),
            d_c_f: v.add(Tensor::new(ctx, vec![b, d], TensorMode::Mut)),

            h_in_t: v.add(Tensor::new(ctx, vec![1+input_size+d, b], TensorMode::Mut)),
            wlstm_t: v.add(Tensor::new(ctx, vec![4*d, 1+input_size+d], TensorMode::Mut)),

            input_size: input_size,
            hidden_size: hidden_size,
        }
    }
}

impl Operation for LstmImpl {
    fn forward(&mut self, ctx: &ga::Context, v: &VarStore, node: &Node) {
        let d = self.hidden_size;
        let input_size = self.input_size;

        let x = &v.get(node.inputs[0]); // input
        let wlstm = &v.get(node.inputs[1]); // all of the weights for all the cells
        let prev_h = &v.get(node.inputs[2]); // Output from last timestep
        let prev_c = &v.get(node.inputs[3]); // C from last timestep

        let ref h_in = v.get(self.h_in);

        let ref ifog = v.get(self.ifog);
        let ref ifog_f = v.get(self.ifog_f);
        let ref c_f = v.get(self.c_f);

        let h = &v.get(node.outputs[0]); // output
        let c = &v.get(node.outputs[1]); // cell

        // NOTE: unless the layer is unrolled, c and prev_c are actually the same underlying buffer.
        // We use different aliases for clarity.
        // NOTE: unless the layer is unrolled, h and prev_h are actually the same underlying buffer.
        // We use different aliases for clarity.

        // Input
        ga::copy_to_slice(ctx, &x.slice(s![..]), &h_in.slice(s![.., 1..input_size+1]));
        ga::copy_to_slice(ctx, &prev_h.slice(s![..]), &h_in.slice(s![.., input_size+1..]));
        // Multiply inputs and weights, and add biases - all in one dot product!
        ga::matmul(ctx, &h_in, &wlstm, &ifog);
        // Compute internal activations
        ga::sigmoid_slice(ctx, &ifog.slice(s![.., ..3*d]), &ifog_f.slice(s![.., ..3*d])); // sigmoids
        ga::tanh_slice(ctx, &ifog.slice(s![.., 3*d..]), &ifog_f.slice(s![.., 3*d..])); // tanh
        // compute the LSTM cell activation
        // NOTE: we're using c_f as a temporary buffer here - we overwrite it later anyway
        // c[t] = ifog_f[t, .., ..d]*ifog_f[t, .., 3*d..] + ifog_f[t, .., d..2*d]*c[t-1]
        ga::multiply_slice(ctx, &ifog_f.slice(s![.., d..2*d]), &prev_c.slice(s![..]), &c_f.slice(s![..]));
        ga::multiply_slice(ctx, &ifog_f.slice(s![.., ..d]), &ifog_f.slice(s![.., 3*d..]), &c.slice(s![..]));
        ga::add(ctx, c, -1, c_f, c);
        // c_f[t] = tanh(c[t])
        ga::tanh(ctx, &c, &c_f);
        ga::multiply_slice(ctx, &ifog_f.slice(s![.., 2*d..3*d]), &c_f.slice(s![..]), &h.slice(s![..]));
    }

    fn backward(&mut self, ctx: &ga::Context, v: &VarStore, node: &Node) {
        let d = self.hidden_size;
        let input_size = self.input_size;

        let wlstm = &v.get(node.inputs[1]); // all of the weights for all the cells
        let prev_c = &v.get(node.inputs[3]); // C from last timestep

        let d_x = &v.get(node.in_grad[0]);
        let d_wlstm = &v.get(node.in_grad[1]);
        let d_prev_h = &v.get(node.in_grad[2]);
        let d_prev_c = &v.get(node.in_grad[3]);

        let ref h_in = v.get(self.h_in);
        let ref d_h_in = v.get(self.d_h_in);

        let ref h_in_t = v.get(self.h_in_t);
        let ref wlstm_t = v.get(self.wlstm_t);

        let ref d_ifog = v.get(self.d_ifog);
        let ref ifog_f = v.get(self.ifog_f);
        let ref d_ifog_f = v.get(self.d_ifog_f);
        let ref d_c_inner = v.get(self.d_c_inner);
        let ref c_f = v.get(self.c_f);
        let ref d_c_f = v.get(self.d_c_f);

        let c = &v.get(node.outputs[1]); // cell

        let d_h = &v.get(node.out_grad[0].get());

        // NOTE: unless the layer is unrolled, d_c and d_prev_c are actually the same underlying
        // buffer. We use different aliases for clarity.
        // NOTE: unless the layer is unrolled, d_h and d_prev_h are actually the same underlying
        // buffer. We use different aliases for clarity.

        //tanhCt = Ct[t]
        //dIFOGf[t,:,2*d:3*d] = tanhCt * dHout[t]
        // XXX
        ga::multiply_slice(ctx, &c_f.slice(s![..]), &d_h.slice(s![..]), &d_ifog_f.slice(s![.., 2*d..3*d]));
        // backprop tanh non-linearity first then continue backprop
        //dC[t] += (1-tanhCt**2) * (IFOGf[t,:,2*d:3*d] * dHout[t])
        ga::dtanh(ctx, c, d_c_inner);
        // XXX
        ga::multiply_slice(ctx, &ifog_f.slice(s![.., 2*d..3*d]), &d_h.slice(s![..]), &d_c_f.slice(s![..]));
        ga::multiply(ctx, d_c_f, -1, d_c_inner, d_c_inner);
        if let Some(d_c) = node.out_grad[1].try_get() {
            ga::add(ctx, &v.get(d_c), -1, d_c_inner, d_c_inner);
        }

        //dIFOGf[t,:,d:2*d] = C[t-1] * dC[t]
        ga::multiply_slice(ctx, &prev_c.slice(s![..]), &d_c_inner.slice(s![..]), &d_ifog_f.slice(s![.., d..2*d]));
        //dC[t-1] += IFOGf[t,:,d:2*d] * dC[t]
        ga::multiply_slice(ctx, &ifog_f.slice(s![.., d..2*d]), &d_c_inner.slice(s![..]), &d_prev_c.slice(s![..]));

        //dIFOGf[t,:,:d] = IFOGf[t,:,3*d:] * dC[t]
        ga::multiply_slice(ctx, &ifog_f.slice(s![.., 3*d..]), &d_c_inner.slice(s![..]), &d_ifog_f.slice(s![.., ..d]));
        //dIFOGf[t,:,3*d:] = IFOGf[t,:,:d] * dC[t]
        ga::multiply_slice(ctx, &ifog_f.slice(s![.., ..d]), &d_c_inner.slice(s![..]), &d_ifog_f.slice(s![.., 3*d..]));

        // backprop activation functions
        //dIFOG[t,:,3*d:] = (1 - IFOGf[t,:,3*d:] ** 2) * dIFOGf[t,:,3*d:]
        ga::dtanh_slice(ctx, &ifog_f.slice(s![.., 3*d..]), &d_ifog.slice(s![.., 3*d..]));
        ga::multiply_slice(ctx, &d_ifog.slice(s![.., 3*d..]), &d_ifog_f.slice(s![.., 3*d..]), &d_ifog.slice(s![.., 3*d..]));
        //y = IFOGf[t,:,:3*d]
        //dIFOG[t,:,:3*d] = (y*(1.0-y)) * dIFOGf[t,:,:3*d]
        ga::dsigmoid_slice(ctx, &ifog_f.slice(s![.., ..3*d]), &d_ifog.slice(s![.., ..3*d]));
        ga::multiply_slice(ctx, &d_ifog.slice(s![.., ..3*d]), &d_ifog_f.slice(s![.., ..3*d]), &d_ifog.slice(s![.., ..3*d]));

        // backprop matrix multiply
        //dWLSTM += np.dot(Hin[t].transpose(), dIFOG[t])

        ga::transpose(ctx, h_in, h_in_t);
        ga::matmul(ctx, h_in_t, d_ifog, d_wlstm);

        //dHin[t] = dIFOG[t].dot(WLSTM.transpose())

        ga::transpose(ctx, wlstm, wlstm_t);
        ga::matmul(ctx, d_ifog, wlstm_t, d_h_in);
        //println!("{:?}", d_wlstm.get(ctx));
        //println!("{:?}", d_h_in.get(ctx));

        // backprop the identity transforms into Hin
        //dX[t] = dHin[t,:,1:input_size+1]
        ga::copy_to_slice(ctx, &d_h_in.slice(s![.., 1..input_size+1]), &d_x.slice(s![..]));
        //dHout[t-1,:] += dHin[t,:,input_size+1:]
        // XXX
        ga::copy_to_slice(ctx, &d_h_in.slice(s![.., input_size+1..]), &d_prev_h.slice(s![..]));
    }

    fn reset_rnn(&mut self, ctx: &ga::Context, v: &VarStore, node: &Node) {
        let prev_h = &v.get(node.inputs[2]); // H from last timestep
        let prev_c = &v.get(node.inputs[3]); // C from last timestep
        ga::fill(ctx, prev_h, 0.0);
        ga::fill(ctx, prev_c, 0.0);
    }
}
