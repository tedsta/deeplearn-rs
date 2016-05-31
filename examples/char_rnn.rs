extern crate deeplearn;
extern crate gpuarray as ga;

use std::fs::File;
use std::io::{
    self,
    BufReader,
    Read,
};
use std::path::Path;
use std::rc::Rc;

use deeplearn::{init, layers, util, Graph, Trainer};
use deeplearn::op::Relu;
use ga::Array;

fn main() {
    let batch_size = 5;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Build the graph

    let ctx = Rc::new(ga::Context::new());
    let ref mut graph = Graph::new(ctx.clone());

    // Input. 1 batch of rows*columns inputs
    let input = graph.add_variable(vec![batch_size, 26], false, 0.0);

    // Layer 1
    // Biased fully connected layer with 300 neurons and ReLU activation
    let (l1_fcb, _, _) = layers::dense_biased(graph, input, 300,
                                              init::Normal(0.001, 0.005),  // Weights initializer
                                              init::Normal(0.001, 0.005)); // Bias initializer
    let l1_out = layers::activation(graph, Relu(l1_fcb));

    // Layer 2
    // LSTM layer with 100 cells
    let (l2_out, _) = layers::lstm(graph, l1_out, 100, init::Normal(0.001, 0.005));

    // Layer 3
    // Biased fully connected layer with 26 neurons and ReLU activation
    let (l3_fcb, _, _) = layers::dense_biased(graph, l2_out, 26,
                                              init::Normal(0.001, 0.005),  // Weights initializer
                                              init::Normal(0.001, 0.005)); // Bias initializer
    let l3_out = layers::activation(graph, Relu(l3_fcb));

    // Loss
    let (loss_out, train_out) = layers::mse(graph, l2_out);
    let loss_d = graph.add_gradient(loss_out); // Create a gradient to apply to the loss function

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Train and validate the network

    // TODO
}
