extern crate deeplearn;
extern crate gpuarray as ga;

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{
    self,
    BufReader,
    Write,
};
use std::path::Path;
use std::rc::Rc;

use deeplearn::{init, layers, util, train, Graph};
use deeplearn::op::Relu;
use ga::Array;

fn main() {
    let batch_size = 1;

    let (char_map, rev_char_map, lines) = load_char_rnn_data("data/bible.txt").unwrap();
    let char_classes = char_map.len();
    println!("Loaded char rnn data");
    println!("Character types: {}", char_classes);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Build the graph

    let ctx = Rc::new(ga::Context::new());
    let ref mut graph = Graph::new(ctx.clone());

    // Input. 1 batch of rows*columns inputs
    let input = graph.add_variable(vec![batch_size, char_classes], false, 0.0);

    // Layer 1
    // LSTM layer with 200 cells
    let (l1_out, _) = layers::lstm(graph, input, 200, init::Normal(0.1, 0.5));

    // Layer 2
    // LSTM layer with 200 cells
    let (l2_out, _) = layers::lstm(graph, l1_out, 200, init::Normal(0.1, 0.5));

    // Layer 3
    // Biased fully connected layer with 26 neurons and ReLU activation
    let (l3_fcb, _, _) = layers::dense_biased(graph, l2_out, char_classes,
                                              init::Normal(0.001, 0.005),  // Weights initializer
                                              init::Normal(0.001, 0.005)); // Bias initializer
    let l3_out = layers::activation(graph, Relu(l3_fcb));

    // Loss
    let (loss_out, train_out) = layers::mse(graph, l3_out);
    let loss_d = graph.add_gradient(loss_out); // Create a gradient to apply to the loss function

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Train and validate the network

    // We apply a gradient of -0.001 to the loss function
    let loss_d_cpu = Array::new(vec![batch_size, char_classes], -0.01);
    loss_d.write(graph, &loss_d_cpu);

    let samples = 500;
    let mut i = 0;
    for line in &lines {
        for t in 1..line.len() {
            input.write(graph, &char_map[&line[t-1]]);
            train_out.write(graph, &char_map[&line[t]]);
            graph.forward_rnn(t-1);
        }
        for t in (1..line.len()).rev() {
            graph.backward_rnn(t-1);
        }

        graph.reset_rnn();
        train::apply_gradients(graph);

        println!("{}", i);

        i += 1;
        if i > samples {
            break;
        }
    }

    let mut l3_out_cpu = Array::new(vec![batch_size, char_classes], 0.0);
    let mut argmax_out = Array::new(vec![batch_size], 0usize);

    loop {
        // Print prompt
        print!(">");
        io::stdout().flush().unwrap();

        // Get seed string
        let mut seed = String::new();
        io::stdin().read_line(&mut seed).unwrap();

        // Input the seed string
        let mut last_char = 0u8;
        for c in seed.trim_right().as_bytes() {
            print!("{}", *c as char);
            input.write(graph, &char_map[c]);
            graph.forward();
            l3_out.read(graph, &mut l3_out_cpu);
            util::argmax_rows(&l3_out_cpu, &mut argmax_out);
            last_char = rev_char_map[&argmax_out[&[0]]];
        }

        // Generate the rest of the output
        for _ in 0..150 {
            print!("{}", last_char as char);
            io::stdout().flush().unwrap();

            input.write(graph, &char_map[&last_char]);
            graph.forward();
            l3_out.read(graph, &mut l3_out_cpu);
            //println!("{:?}", l3_out_cpu);
            util::argmax_rows(&l3_out_cpu, &mut argmax_out);
            last_char = rev_char_map[&argmax_out[&[0]]];

            if last_char == b'\n' {
                break;
            }
        }
        println!("");
    }
}

pub fn load_char_rnn_data<P: AsRef<Path>>(path: P)
    -> io::Result<(HashMap<u8, Array<f32>>, HashMap<usize, u8>, Vec<Vec<u8>>)>
{
    use std::io::BufRead;

    let ref mut file = BufReader::new(File::open(path).unwrap());

    let mut unique_chars = HashSet::new();
    let mut lines = vec![];

    for line in file.lines() {
        let line: String = try!(line) + "\n";
        for c in line.as_bytes() {
            unique_chars.insert(*c);
        }
        lines.push(line.as_bytes().to_owned());
    }

    let char_classes = unique_chars.len();
    let mut char_map = HashMap::new();
    let mut rev_char_map = HashMap::new();
    for (i, c) in unique_chars.into_iter().enumerate() {
        char_map.insert(c, util::one_hot_row(i, char_classes));
        rev_char_map.insert(i, c);
    }

    Ok((char_map, rev_char_map, lines))
}
