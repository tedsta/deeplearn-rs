use graph::Graph;
use var_store::VarIndex;

use ga::{self, Array, Tensor, TensorMode};

pub struct Trainer;

impl Trainer {
    pub fn new() -> Trainer {
        Trainer
    }

    pub fn train<O: Optimizer, F>(&self, graph: &mut Graph, optimizer: &O, epochs: usize,
                                  mut update_fn: F, training_data: &[(VarIndex, &[Array<f32>])])
                                  where F: FnMut(&mut Graph, usize)
    {
        for epoch in 0..epochs {
            // Upload training data
            for &(var, data) in training_data {
                var.write(graph, &data[epoch]);
            }

            // Run the graph
            graph.forward();
            graph.backward();

            optimizer.update(graph);

            update_fn(graph, epoch);
        }
    }
}

pub trait Optimizer {
    fn update(&self, graph: &Graph);
}

/// Stochastic gradient descent
pub struct Sgd {
    learn_rate: f32,
}

impl Sgd {
    pub fn new(learn_rate: f32) -> Sgd {
        Sgd {
            learn_rate: learn_rate,
        }
    }
}

impl Optimizer for Sgd {
    fn update(&self, graph: &Graph) {
        for &(learn, learn_d) in graph.learnables().iter() {
            ga::sgd(graph.context(), &learn.get(graph), &learn_d.get(graph), self.learn_rate);
        }
    }
}

pub struct RmsProp {
    cache: Vec<Tensor<f32>>, // gradient cache
    learn_rate: f32,
    decay_rate: f32,
}

impl RmsProp {
    pub fn new(graph: &Graph, learn_rate: f32, decay_rate: f32) -> RmsProp {
        let cache = graph.learnables().iter().map(
            |&(_, learn_d)| {
                Tensor::new(graph.context(), learn_d.get(graph).shape().to_owned(), TensorMode::Mut)
            }).collect();
        RmsProp {
            cache: cache,
            learn_rate: learn_rate,
            decay_rate: decay_rate,
        }
    }
}

impl Optimizer for RmsProp {
    fn update(&self, graph: &Graph) {
        for (&(learn, learn_d), cache) in graph.learnables().iter().zip(self.cache.iter()) {
            ga::rmsprop(graph.context(), &learn.get(graph), &learn_d.get(graph), cache, self.learn_rate, self.decay_rate, 0.00001);
        }
    }
}
