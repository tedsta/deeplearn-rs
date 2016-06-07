use graph::Graph;
use var_store::VarIndex;

use ga::{self, Array, Tensor, TensorMode};

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

            apply_gradients(graph);

            update_fn(graph, epoch);
        }
    }
}

pub struct RmsPropTrainer {
    cache: Vec<Tensor<f32>>, // gradient cache
}

impl RmsPropTrainer {
    pub fn new(graph: &Graph) -> RmsPropTrainer {
        let cache = graph.learnables().iter().map(
            |&(_, learn_d)| {
                Tensor::new(graph.context(), learn_d.get(graph).shape().to_owned(), TensorMode::Mut)
            }).collect();
        RmsPropTrainer {
            cache: cache,
        }
    }
}

pub fn apply_gradients(graph: &Graph) {
    // Apply gradients
    for &(learn, learn_d) in graph.learnables().iter() {
        //println!("{:?}", learn_d.get(graph).get(graph.context()));
        ga::add(graph.context(), &learn.get(graph), -1, &learn_d.get(graph), &learn.get(graph));
    }
}
