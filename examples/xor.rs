extern crate deeplearn;
extern crate matrix;

use deeplearn::Graph;
use deeplearn::op::{MatMul, Relu};

fn main() {
    let ctx = matrix::Context::new();

    // Setup the graph
    let mut graph = Graph::new();
    let a = graph.add_variable(&ctx, (1, 2));
    let l1_w = graph.add_variable(&ctx, (2, 3));
    let l1_mat_mul = graph.add_node(&ctx,
                                    Box::new(MatMul::new(&ctx, (1, 2), (2, 3))),
                                    vec![a, l1_w],
                                    &[(1, 3)]);
    let l1_mat_mul_out = l1_mat_mul.get(&graph).outputs[0];
    let l1_relu = graph.add_node(&ctx,
                                 Box::new(Relu::new(&ctx, (1, 3))),
                                 vec![l1_mat_mul_out],
                                 &[(1, 3)]);
    let l1_relu_out = l1_relu.get(&graph).outputs[0];
    let l2_w = graph.add_variable(&ctx, (3, 1));
    let l2_mat_mul = graph.add_node(&ctx,
                                    Box::new(MatMul::new(&ctx, (1, 3), (3, 1))),
                                    vec![l1_relu_out, l2_w],
                                    &[(1, 1)]);
    let l2_mat_mul_out = l2_mat_mul.get(&graph).outputs[0];
    let l2_relu = graph.add_node(&ctx,
                                 Box::new(Relu::new(&ctx, (1, 1))),
                                 vec![l2_mat_mul_out],
                                 &[(1, 1)]);
    let l2_relu_d = graph.add_gradient(&ctx, l2_relu, 0);

    // Send some input data
    let a_cpu = matrix::Matrix::from_vec(1, 2, vec![1.0, -0.3]);
    let l1_w_cpu = matrix::Matrix::from_vec(2, 3, vec![0.5, 0.3, 0.2,
                                                       0.6, 0.7, 0.7]);
    let l2_w_cpu = matrix::Matrix::from_vec(3, 1, vec![0.5, 0.3, 0.2]);
    let l2_relu_d_cpu = matrix::Matrix::from_vec(1, 1, vec![1.0]);
    a.get(&graph).set(&ctx, &a_cpu);
    l1_w.get(&graph).set(&ctx, &l1_w_cpu);
    l2_w.get(&graph).set(&ctx, &l2_w_cpu);
    l2_relu_d.get(&graph).set(&ctx, &l2_relu_d_cpu);

    // Run the network
    graph.run(&ctx);
    let out = l2_relu.get(&graph).outputs[0].get(&graph).get(&ctx);
    let l1_w_d = graph.get_input_gradient(l1_w).unwrap().get(&graph).get(&ctx);
    println!("out = {:?}", out);
    println!("l1_w_d = {:?}", l1_w_d);
}
