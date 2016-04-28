use graph::Graph;
use var_store::VarIndex;

use ga::{self, Array};

pub struct Trainer;

impl Trainer {
    pub fn new() -> Trainer {
        Trainer
    }

    pub fn train(&self, graph: &mut Graph, epochs: usize,
                 learn_vars: Vec<VarIndex>,
                 out_vars: Vec<VarIndex>,
                 train_in: Vec<(VarIndex, &Vec<Array<f32>>)>,
                 train_out: Vec<(VarIndex, &Vec<Array<f32>>)>) {
        // Create CPU-side arrays to download our outputs into
        let mut outputs: Vec<Array<f32>> =
            out_vars.iter()
                    .map(|out| Array::new(out.get(&graph).shape().to_owned(), 0.0))
                    .collect();

        for epoch in 0..epochs {
            // Upload inputs
            for &(t_in_var, t_in) in &train_in {
                t_in_var.get(&graph).set(graph.context(), t_in.get(epoch).unwrap());
            }

            // Upload training labels
            for &(t_out_var, t_out) in &train_out {
                t_out_var.get(&graph).set(graph.context(), t_out.get(epoch).unwrap());
            }

            // Run the graph
            graph.forward();
            graph.backward();

            // Get the output
            for (out_var, output) in out_vars.iter().zip(outputs.iter_mut()) {
                out_var.get(&graph).read(&graph.context(), output);
            }

            // Apply gradients
            for learn in learn_vars.iter().cloned() {
                let learn_d = graph.get_input_gradient(learn).unwrap().get(&graph);
                ga::add(&graph.context(), &learn.get(&graph), -1, &learn_d, &learn.get(&graph));
            }

            if epoch % 1000 == 999 {
                println!("===================");
                println!("epoch {}", epoch);
                println!("out =\n{:?}\n", outputs);
            }
        }
    }
}
