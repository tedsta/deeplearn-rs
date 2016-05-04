use graph::Graph;
use var_store::VarIndex;

use ga::{self, Array};

pub struct Trainer;

impl Trainer {
    pub fn new() -> Trainer {
        Trainer
    }

    pub fn train<F>(&self, graph: &mut Graph, epochs: usize, mut update_fn: F,
                    out_vars: &[VarIndex],
                    training_data: &[(VarIndex, &[Array<f32>])])
                    where F: FnMut(&mut Graph, usize),
    {
        // Create CPU-side arrays to download our outputs into
        let mut outputs: Vec<Array<f32>> =
            out_vars.iter()
                    .map(|out| Array::new(out.get(graph).shape().to_owned(), 0.0))
                    .collect();

        for epoch in 0..epochs {
            // Upload training data
            for &(var, data) in training_data {
                var.get(graph).set(graph.context(), data.get(epoch).unwrap());
            }

            // Run the graph
            graph.forward();
            graph.backward();

            // Get the output
            for (out_var, output) in out_vars.iter().zip(outputs.iter_mut()) {
                out_var.get(graph).read(graph.context(), output);
            }

            // Apply gradients
            for &(learn, learn_d) in graph.learnables().iter() {
                ga::add(graph.context(), &learn.get(graph), -1, &learn_d.get(graph), &learn.get(graph));
            }

            update_fn(graph, epoch);
        }
    }
}
