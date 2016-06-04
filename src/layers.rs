use graph::Graph;
use init::Initializer;
use op::{Add, CrossEntropy, Lstm, LstmUnrolled, MatMul, Mse, OpBuilder};
use var_store::VarIndex;

pub fn dense<WI: Initializer>(graph: &mut Graph,
                              input: VarIndex,
                              layer_size: usize,
                              w_init: WI)
                              -> (VarIndex, VarIndex) {
    // Input shape is [batch_size x input_size]
    let input_size = input.get(graph).shape()[1];

    // Weights for layer 1: [input_size x layer_size]
    let weights = graph.add_variable(vec![input_size, layer_size], true, w_init);

    // Use matrix multiplication to do a fully connected layer
    let mat_mul = graph.add_node(MatMul(input, weights));
    let mat_mul_out = mat_mul.get(&graph).outputs[0]; 

    (mat_mul_out, weights)
}

pub fn dense_biased<WI: Initializer, BI: Initializer>(graph: &mut Graph,
                                                      input: VarIndex,
                                                      layer_size: usize,
                                                      w_init: WI,
                                                      b_init: BI)
                                                      -> (VarIndex, VarIndex, VarIndex) {
    // Input shape is [batch_size x input_size]
    let input_size = input.get(graph).shape()[1];

    // Weights for layer 1: [input_size x layer_size]
    let weights = graph.add_variable(vec![input_size, layer_size], true, w_init);

    // Use matrix multiplication to do a fully connected layer
    let mat_mul = graph.add_node(MatMul(input, weights));
    let mat_mul_out = mat_mul.get(&graph).outputs[0]; 

    // Biases, one for each neuron in layer
    let bias = graph.add_variable(vec![1, layer_size], true, b_init);
    // Add the biases to the matrix multiplication output
    let biased = graph.add_node(Add(mat_mul_out, bias, 0));
    // Grab VarIndex for biased's output
    let biased_out = biased.get(&graph).outputs[0];

    (biased_out, weights, bias)
}

pub fn dense_biased_manual(graph: &mut Graph,
                           input: VarIndex,
                           weights: VarIndex,
                           bias: VarIndex)
                           -> VarIndex {
    // Use matrix multiplication to do a fully connected layer
    let mat_mul = graph.add_node(MatMul(input, weights));
    let mat_mul_out = mat_mul.get(&graph).outputs[0]; 

    // Add the biases to the matrix multiplication output
    let biased = graph.add_node(Add(mat_mul_out, bias, 0));
    // Grab VarIndex for biased's output
    let biased_out = biased.get(&graph).outputs[0];

    biased_out
}

pub fn activation<A: OpBuilder>(graph: &mut Graph, op: A) -> VarIndex {
    // Run the biased input*weight sums through an ReLU activation
    let activation = graph.add_node(op);
    // Grab VarIndex for l2_relu's output
    let activation_out = activation.get(&graph).outputs[0];

    activation_out
}

pub fn lstm<WI: Initializer>(graph: &mut Graph,
                             input: VarIndex,
                             layer_size: usize,
                             w_init: WI)
                             -> (VarIndex, VarIndex) {
    // Input shape is [batch_size, input_size]
    let input_size = input.get(graph).shape()[1];

    // Weights for layer 1: [1+input_size+layer_size, 4*layer_size]
    let weights = graph.add_variable(vec![1+input_size+layer_size, 4*layer_size], true, w_init);

    // Use matrix multiplication to do a fully connected layer
    let lstm = graph.add_node(Lstm(input, weights, layer_size));
    let lstm_out = lstm.get(&graph).outputs[0]; 

    (lstm_out, weights)
}

pub fn lstm_unrolled(graph: &mut Graph,
                     input: VarIndex,
                     weights: VarIndex,
                     prev_h: VarIndex,
                     prev_c: VarIndex)
                     -> (VarIndex, VarIndex) {
    // Use matrix multiplication to do a fully connected layer
    let lstm = graph.add_node(LstmUnrolled(input, weights, prev_h, prev_c));
    let lstm_out = lstm.get(&graph).outputs[0]; 
    let c = lstm.get(&graph).outputs[1]; 

    (lstm_out, c)
}

pub fn mse(graph: &mut Graph, out: VarIndex) -> (VarIndex, VarIndex) {
    let out_shape = out.get(graph).shape().to_owned();

    // Expected output
    let train_out = graph.add_variable(out_shape, false, 0.0);

    let loss = graph.add_node(Mse(out, train_out));
    let loss_out = loss.get(&graph).outputs[0];

    (loss_out, train_out)
}

pub fn cross_entropy(graph: &mut Graph, out: VarIndex) -> (VarIndex, VarIndex) {
    let out_shape = out.get(graph).shape().to_owned();

    // Expected output
    let train_out = graph.add_variable(out_shape, false, 0.0);

    let loss = graph.add_node(CrossEntropy(out, train_out));
    let loss_out = loss.get(&graph).outputs[0];

    (loss_out, train_out)
}
