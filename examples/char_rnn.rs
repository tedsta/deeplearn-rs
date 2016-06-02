extern crate deeplearn;
extern crate gpuarray as ga;

use std::collections::{HashMap, HashSet};
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
    let batch_size = 1;

    let (char_map, lines) = load_char_rnn_data("data/bible.txt").unwrap();
    println!("Loaded char rnn data");

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

pub fn load_char_rnn_data<P: AsRef<Path>>(path: P)
    -> io::Result<(HashMap<u8, Array<f32>>, Vec<Vec<u8>>)>
{
    use std::io::BufRead;

    let ref mut file = BufReader::new(File::open(path).unwrap());

    let mut unique_chars = HashSet::new();
    let mut lines = vec![];

    for line in file.lines() {
        let line: String = try!(line);
        for c in line.as_bytes() {
            unique_chars.insert(*c);
        }
        lines.push(line.as_bytes().to_owned());
    }

    let char_classes = unique_chars.len();
    let mut char_map = HashMap::new();
    for (i, c) in unique_chars.into_iter().enumerate() {
        char_map.insert(c, util::one_hot_row(i, char_classes));
    }

    Ok((char_map, lines))
}
