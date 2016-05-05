use graph::Graph;
use var_store::VarIndex;

use ga::{self, Array};

pub struct Trainer;

impl Trainer {
    pub fn new() -> Trainer {
        Trainer
    }

    pub fn train<F>(&self, graph: &mut Graph, epochs: usize, mut update_fn: F,
                    training_data: &[(VarIndex, &[Array<f32>])])
                    where F: FnMut(&mut Graph, usize),
    {
        for epoch in 0..epochs {
            // Upload training data
            for &(var, data) in training_data {
                var.write(graph, &data[epoch]);
            }

            // Run the graph
            graph.forward();
            graph.backward();

            // Apply gradients
            for &(learn, learn_d) in graph.learnables().iter() {
                ga::add(graph.context(), &learn.get(graph), -1, &learn_d.get(graph), &learn.get(graph));
            }

            update_fn(graph, epoch);
        }
    }
}
