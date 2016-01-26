extern crate deeplearn;
extern crate matrix;

use deeplearn::Graph;
use deeplearn::op::{MatMul, Mse, Relu};

fn main() {
    let ctx = matrix::Context::new();

    // Setup the graph
    let mut graph = Graph::new();

    // Input
    let a = graph.add_variable(&ctx, (5, 2));
    // Layer 1
    let l1_w = graph.add_variable(&ctx, (2, 3));
    let l1_mat_mul = graph.add_node(&ctx,
                                    Box::new(MatMul::new(&ctx, (5, 2), (2, 3))),
                                    vec![a, l1_w],
                                    &[(5, 3)]);
    let l1_mat_mul_out = l1_mat_mul.get(&graph).outputs[0];
    let l1_relu = graph.add_node(&ctx,
                                 Box::new(Relu::new(&ctx, (5, 3))),
                                 vec![l1_mat_mul_out],
                                 &[(5, 3)]);
    let l1_relu_out = l1_relu.get(&graph).outputs[0];
    // Layer 2
    let l2_w = graph.add_variable(&ctx, (3, 1));
    let l2_mat_mul = graph.add_node(&ctx,
                                    Box::new(MatMul::new(&ctx, (5, 3), (3, 1))),
                                    vec![l1_relu_out, l2_w],
                                    &[(5, 1)]);
    let l2_mat_mul_out = l2_mat_mul.get(&graph).outputs[0];
    let l2_relu = graph.add_node(&ctx,
                                 Box::new(Relu::new(&ctx, (5, 1))),
                                 vec![l2_mat_mul_out],
                                 &[(5, 1)]);
    let l2_relu_out = l2_relu.get(&graph).outputs[0];
    // Loss
    let train_out = graph.add_variable(&ctx, (5, 1));
    let loss = graph.add_node(&ctx,
                              Box::new(Mse::new(&ctx, (5, 1))),
                              vec![l2_relu_out, train_out],
                              &[(1, 1)]);
    let loss_out = loss.get(&graph).outputs[0];
    let loss_d = graph.add_gradient(&ctx, loss, 0);

    // Send some input data
    let a_cpu = matrix::Matrix::from_vec(5, 2, vec![1.0, 0.0,
                                                    0.0, 1.0,
                                                    0.0, 0.0,
                                                    1.0, 1.0,
                                                    0.0, 0.5]);
    let train_out_cpu = matrix::Matrix::from_vec(5, 1, vec![1.0, 1.0, 0.0, 0.0, 0.5]);
    let l1_w_cpu = matrix::Matrix::from_vec(2, 3, vec![0.5, 0.3, 0.2,
                                                       0.6, 0.7, 0.7]);
    let l2_w_cpu = matrix::Matrix::from_vec(3, 1, vec![0.5, 0.3, 0.2]);
    let loss_d_cpu = matrix::Matrix::from_vec(1, 1, vec![1.0]);
    a.get(&graph).set(&ctx, &a_cpu);
    train_out.get(&graph).set(&ctx, &train_out_cpu);
    l1_w.get(&graph).set(&ctx, &l1_w_cpu);
    l2_w.get(&graph).set(&ctx, &l2_w_cpu);
    loss_d.get(&graph).set(&ctx, &loss_d_cpu);

    // Run/Train the network
    for epoch in 0..1000 {
        graph.run(&ctx);
        let out = l2_relu_out.get(&graph).get(&ctx);
        let out_d = l2_relu.get(&graph)
                           .out_grad[0]
                           .gradient()
                           .get(&graph)
                           .get(&ctx);
        let l = loss_out.get(&graph).get(&ctx);
        let l1_w_d = graph.get_input_gradient(l1_w).unwrap().get(&graph).get(&ctx);
        let l2_w_d = graph.get_input_gradient(l1_w).unwrap().get(&graph).get(&ctx);
        if epoch % 1000 == 0 {
            println!("===================");
            println!("Epoch: {}", epoch);
            println!("out = {:?}", out);
            println!("out_d = {:?}", out_d);
            println!("loss = {:?}", l);
            println!("l1_w_d = {:?}", l1_w_d);
            println!("l2_w_d = {:?}", l2_w_d);
        }
    }
}
