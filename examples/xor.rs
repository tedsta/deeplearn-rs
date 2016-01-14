extern crate deeplearn;
extern crate matrix;

use deeplearn::Graph;
use deeplearn::op::{MatMul, Relu};

fn main() {
    let ctx = matrix::Context::new();

    // Setup the graph
    let mut graph = Graph::new();
    let a = graph.add_variable(&ctx, (1, 2));
    let wa = graph.add_variable(&ctx, (2, 3));
    let mat_mul = graph.add_node(&ctx,
                                 Box::new(MatMul::new(&ctx, (1, 2), (2, 3))),
                                 vec![a, wa],
                                 &[(1, 3)]);
    let mat_mul_out = mat_mul.get(&graph).outputs[0];
    let relu = graph.add_node(&ctx,
                              Box::new(Relu::new(&ctx, (1, 3))),
                              vec![mat_mul_out],
                              &[(1, 3)]);
    let relu_d = graph.add_gradient(&ctx, relu, 0);

    // Send some input data
    let a_cpu = matrix::Matrix::from_vec(1, 2, vec![1.0, -0.3]);
    let wa_cpu = matrix::Matrix::from_vec(2, 3, vec![0.5, 0.3, 0.2,
                                                     0.6, 0.7, 0.7]);
    let relu_d_cpu = matrix::Matrix::from_vec(1, 3, vec![1.0, 1.0, 1.0]);
    a.get(&graph).set(&ctx, &a_cpu);
    wa.get(&graph).set(&ctx, &wa_cpu);
    relu_d.get(&graph).set(&ctx, &relu_d_cpu);

    // Run the network
    graph.run(&ctx);
    let out = relu.get(&graph).outputs[0].get(&graph).get(&ctx);
    let wa_d = graph.get_input_gradient(wa).unwrap().get(&graph).get(&ctx);
    println!("out = {:?}", out);
    println!("wa_d = {:?}", wa_d);
}
