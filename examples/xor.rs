extern crate deeplearn;
extern crate matrix;

use deeplearn::Graph;
use deeplearn::op::{Add, MatMul, Mse, Relu};

fn main() {
    let ctx = matrix::Context::new();

    // Setup the graph
    let mut graph = Graph::new();

    // Input
    let a = graph.add_variable(&ctx, (5, 2));
    // Layer 1
    let l1_w = graph.add_variable(&ctx, (2, 6));
    let l1_mat_mul = graph.add_node(&ctx,
                                    Box::new(MatMul::new(&ctx, (5, 2), (2, 6))),
                                    vec![a, l1_w],
                                    &[(5, 6)]);
    let l1_mat_mul_out = l1_mat_mul.get(&graph).outputs[0];
    let l1_b = graph.add_variable(&ctx, (1, 6));
    let l1_fc = graph.add_node(&ctx,
                               Box::new(Add::new(0)),
                               vec![l1_b, l1_mat_mul_out],
                               &[(5, 6)]);
    let l1_fc_out = l1_fc.get(&graph).outputs[0];
    /*let l1_relu = graph.add_node(&ctx,
                                 Box::new(Relu::new()),
                                 vec![l1_fc_out],
                                 &[(5, 3)]);
    let l1_relu_out = l1_relu.get(&graph).outputs[0];*/
    // Layer 2
    let l2_w = graph.add_variable(&ctx, (6, 1));
    let l2_mat_mul = graph.add_node(&ctx,
                                    Box::new(MatMul::new(&ctx, (5, 6), (6, 1))),
                                    vec![l1_fc_out, l2_w],
                                    &[(5, 1)]);
    let l2_mat_mul_out = l2_mat_mul.get(&graph).outputs[0];
    let l2_b = graph.add_variable(&ctx, (5, 1));
    let l2_fc = graph.add_node(&ctx,
                               Box::new(Add::new(0)),
                               vec![l2_b, l2_mat_mul_out],
                               &[(5, 1)]);
    let l2_fc_out = l2_fc.get(&graph).outputs[0];
    let l2_relu = graph.add_node(&ctx,
                                 Box::new(Relu::new()),
                                 vec![l2_fc_out],
                                 &[(5, 1)]);
    let l2_relu_out = l2_relu.get(&graph).outputs[0];
    // Loss
    let train_out = graph.add_variable(&ctx, (5, 1));
    let loss = graph.add_node(&ctx,
                              Box::new(Mse::new()),
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
    let l1_w_cpu = matrix::Matrix::from_vec(2, 6, vec![0.5, 0.3, 0.2, 0.5, -0.3, 0.2,
                                                       0.6, 0.7, 0.7, 0.6, 0.7, 0.7]);
    let l1_b_cpu = matrix::Matrix::from_vec(1, 6, vec![0.3, 1.0, 0.7, -0.1, 0.4, 0.1]);
    let l2_w_cpu = matrix::Matrix::from_vec(6, 1, vec![0.5, -0.3, 0.2, -0.1, 0.4, 0.1]);
    let l2_b_cpu = matrix::Matrix::from_vec(1, 1, vec![0.8]);
    let loss_d_cpu = matrix::Matrix::from_vec(1, 1, vec![-0.4]);
    a.get(&graph).set(&ctx, &a_cpu);
    train_out.get(&graph).set(&ctx, &train_out_cpu);
    l1_w.get(&graph).set(&ctx, &l1_w_cpu);
    l1_b.get(&graph).set(&ctx, &l1_b_cpu);
    l2_w.get(&graph).set(&ctx, &l2_w_cpu);
    l2_b.get(&graph).set(&ctx, &l2_b_cpu);
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
        let l1_w_d = graph.get_input_gradient(l1_w).unwrap().get(&graph);
        let l2_w_d = graph.get_input_gradient(l1_w).unwrap().get(&graph);
        l1_w.get(&graph).add(&ctx, -1, &*l1_w_d, &*l1_w.get(&graph));
        l2_w.get(&graph).add(&ctx, -1, &*l2_w_d, &*l2_w.get(&graph));
        if epoch % 100 == 0 {
            println!("===================");
            println!("Epoch: {}", epoch);
            println!("out = {:?}", out);
            println!("out_d = {:?}", out_d);
            println!("loss = {:?}", l);
            println!("l1_w = {:?}", l1_w.get(&graph).get(&ctx));
            println!("l2_w = {:?}", l2_w.get(&graph).get(&ctx));
        }
    }
}
