extern crate deeplearn;
extern crate matrix;

use deeplearn::Graph;
use deeplearn::op::{Add, MatMul, Mse, Relu};
use matrix::Matrix;

fn main() {
    let ctx = matrix::Context::new();

    // Setup the graph. It's going to have 2 inputs, 3 hidden nodes in it's 1 hidden layer, and 1
    // output node
    let mut graph = Graph::new();

    //////////////////////////
    // Layer 1

    // Input to xor gate. 5 batches of 2 inputs each.
    let a = graph.add_variable(&ctx, (5, 2));
    // Weights for layer 1: [2 inputs x 3 nodes]
    let l1_w = graph.add_variable(&ctx, (2, 3));
    // Use matrix multiplication to do a fully connected layer
    let l1_mat_mul = graph.add_node(&ctx,
                                    MatMul::new(&ctx, (5, 2), (2, 3)),
                                    vec![a, l1_w],
                                    &[(5, 3)]); // out shape: [5x2]*[2x3] = [5 batches x 3 outputs]
    // Grab VarIndex for l1_mat_mul's output
    let l1_mat_mul_out = l1_mat_mul.get(&graph).outputs[0]; 
    // 3 biases; one for each node in layer 1
    let l1_b = graph.add_variable(&ctx, (1, 3));
    // Here we add the biases to the matrix multiplication output
    let l1_biased = graph.add_node(&ctx,
                               Add::new(0),
                               vec![l1_mat_mul_out, l1_b],
                               &[(5, 3)]);
    // Grab VarIndex for l1_biased's output
    let l1_biased_out = l1_biased.get(&graph).outputs[0];
    // Run the biased input*weight sums through an ReLU activation
    let l1_relu = graph.add_node(&ctx,
                                 Relu::new(),
                                 vec![l1_biased_out],
                                 &[(5, 3)]);
    let l1_relu_out = l1_relu.get(&graph).outputs[0];

    //////////////////////////
    // Layer 2

    // Weights for layer 2: [3 inputs x 1 nodes]
    let l2_w = graph.add_variable(&ctx, (3, 1));
    // Fully connected layer 2. Use layer 1's output as layer 2's input
    let l2_mat_mul = graph.add_node(&ctx,
                                    MatMul::new(&ctx, (5, 3), (3, 1)),
                                    vec![l1_relu_out, l2_w],
                                    &[(5, 1)]); // out shape: [5x3]*[3x1] = [5 batches x 1 output]
    // Grab VarIndex for l2_mat_mul's output
    let l2_mat_mul_out = l2_mat_mul.get(&graph).outputs[0];
    // 1 bias for 1 output node
    let l2_b = graph.add_variable(&ctx, (1, 1));
    // Here we add the bias to the matrix multiplication output
    let l2_biased = graph.add_node(&ctx,
                               Add::new(0),
                               vec![l2_mat_mul_out, l2_b],
                               &[(5, 1)]);
    // Grab VarIndex for l2_biased's output
    let l2_biased_out = l2_biased.get(&graph).outputs[0];
    // Run the biased input*weight sums through an ReLU activation
    let l2_relu = graph.add_node(&ctx,
                                 Relu::new(),
                                 vec![l2_biased_out],
                                 &[(5, 1)]);
    // Grab VarIndex for l2_relu's output
    let l2_relu_out = l2_relu.get(&graph).outputs[0];

    //////////////////////////
    // Loss

    // Add a variable for the training outputs: [5 batches x 1 output]
    let train_out = graph.add_variable(&ctx, (5, 1));
    // Use mean squared error loss function
    let loss = graph.add_node(&ctx,
                              Mse::new(),
                              vec![l2_relu_out, train_out],
                              &[(1, 1)]);
    // Grab the VarIndex for loss's output
    let loss_out = loss.get(&graph).outputs[0];
    // Create a gradient to apply to the loss function
    let loss_d = graph.add_gradient(&ctx, loss, 0);

    ////////////////////////////////////////////////////////////////////////////////////////////////

    // Send some input data
    let a1_cpu = Matrix::from_vec(5, 2, vec![1.0, 0.0,
                                             0.0, 1.0,
                                             1.0, 0.0,
                                             0.0, 1.0,
                                             0.5, 0.0]);
    let a2_cpu = Matrix::from_vec(5, 2, vec![1.0, 1.0,
                                             1.0, 1.0,
                                             0.0, 0.0,
                                             0.0, 0.0,
                                             0.5, 0.5]);
    let train_out1_cpu = Matrix::from_vec(5, 1, vec![1.0, 1.0, 1.0, 1.0, 0.5]);
    let train_out2_cpu = Matrix::from_vec(5, 1, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    let l1_w_cpu = Matrix::from_vec(2, 3, vec![1.0, 0.3, 0.0,
                                               0.0, 0.8, 0.6]);
    let l1_b_cpu = Matrix::from_vec(1, 3, vec![0.3, -0.5, 0.1]);
    let l2_w_cpu = Matrix::from_vec(3, 1, vec![1.0, 0.5, 1.0]);
    let l2_b_cpu = Matrix::from_vec(1, 1, vec![0.2]);
    // We apply a gradient of -0.1 to the loss function
    let loss_d_cpu = Matrix::from_vec(1, 1, vec![-0.1]);

    // Upload all the data to the gpu
    l1_w.get(&graph).set(&ctx, &l1_w_cpu);
    l1_b.get(&graph).set(&ctx, &l1_b_cpu);
    l2_w.get(&graph).set(&ctx, &l2_w_cpu);
    l2_b.get(&graph).set(&ctx, &l2_b_cpu);
    loss_d.get(&graph).set(&ctx, &loss_d_cpu);

    /////////////////////////
    // Run/Train the network
    for epoch in 0..1000 {
        if epoch % 2 == 0 {
            a.get(&graph).set(&ctx, &a1_cpu);
            train_out.get(&graph).set(&ctx, &train_out1_cpu);
        } else {
            a.get(&graph).set(&ctx, &a2_cpu);
            train_out.get(&graph).set(&ctx, &train_out2_cpu);
        }
        graph.run(&ctx);
        let out = l2_relu_out.get(&graph).get(&ctx);
        let out_d = l2_relu.get(&graph)
                           .out_grad[0]
                           .gradient()
                           .get(&graph)
                           .get(&ctx);
        let l = loss_out.get(&graph).get(&ctx);
        let l1_w_d = graph.get_input_gradient(l1_w).unwrap().get(&graph);
        let l2_w_d = graph.get_input_gradient(l1_w).unwrap().get(&graph);
        l1_w.get(&graph).add(&ctx, -1, &*l1_w_d, &*l1_w.get(&graph));
        l2_w.get(&graph).add(&ctx, -1, &*l2_w_d, &*l2_w.get(&graph));
        if epoch % 100 == 0 || epoch % 100 == 1 {
            println!("===================");
            println!("Epoch: {}", epoch);
            println!("out = {:?}", out);
            println!("out_d = {:?}", out_d);
            println!("loss = {:?}", l);
            println!("l1_w = {:?}", l1_w.get(&graph).get(&ctx));
            println!("l1_b = {:?}", l1_b.get(&graph).get(&ctx));
            println!("l2_w = {:?}", l2_w.get(&graph).get(&ctx));
            println!("l2_b = {:?}", l2_b.get(&graph).get(&ctx));
        }
    }
}
