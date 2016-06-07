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
use deeplearn::op::Softmax;
use ga::Array;

fn main() {
    let batch_size = 1;

    let (char_map, rev_char_map, lines) = load_char_rnn_data("data/bible.txt").unwrap();
    let char_classes = char_map.len();
    println!("Loaded char rnn data");
    println!("Character types: {}", char_classes);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Build the graph

    let batch_size = 1;
    let l1_size = 200;
    let l2_size = 200;

    let ctx = Rc::new(ga::Context::new());
    let ref mut graph = Graph::new(ctx.clone());

    let l1_w = graph.add_variable(vec![1+char_classes+l1_size, 4*l1_size], true, init::Normal(0.001, 0.005));
    let l2_w = graph.add_variable(vec![1+l1_size+l2_size, 4*l2_size], true, init::Normal(0.001, 0.005));
    let l3_w = graph.add_variable(vec![l2_size, char_classes], true, init::Normal(0.001, 0.005));
    let l3_b = graph.add_variable(vec![1, char_classes], true, init::Normal(0.001, 0.005));
    let l1_c0 = graph.add_variable(vec![batch_size, l1_size], false, 0.0);
    let l1_h0 = graph.add_variable(vec![batch_size, l1_size], false, 0.0);
    let l2_c0 = graph.add_variable(vec![batch_size, l2_size], false, 0.0);
    let l2_h0 = graph.add_variable(vec![batch_size, l2_size], false, 0.0);

    let net_step =
        |graph: &mut Graph, (l1_prev_h, l1_prev_c, l2_prev_h, l2_prev_c)| {
            let input = graph.add_variable(vec![batch_size, char_classes], false, 0.0);
            let (l1_out, l1_c) = layers::lstm_unrolled(graph, input, l1_w, l1_prev_h, l1_prev_c);
            let (l2_out, l2_c) = layers::lstm_unrolled(graph, l1_out, l2_w, l2_prev_h, l2_prev_c);
            let l3_fcb = layers::dense_biased_manual(graph, l2_out, l3_w, l3_b);
            let l3_out = layers::activation(graph, Softmax(l3_fcb));
            // Loss
            let (loss_out, train_out) = layers::cross_entropy(graph, l3_out);
            let loss_d = graph.add_gradient(loss_out); // Create a gradient to apply to the loss function
            // We apply a gradient of -0.1 to the loss function
            loss_d.write(graph, &Array::new(loss_d.get(graph).shape().to_owned(), -0.01));

            ((l1_out, l1_c, l2_out, l2_c), (input, l3_out, train_out))
        };

    let (last_recur_in, steps) = util::unrolled_net(graph, 25, (l1_h0, l1_c0, l2_h0, l2_c0), net_step);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Train and validate the network

    let mut l2_out_cpu = Array::new(vec![batch_size, l2_size], 0.0);
    let mut l3_out_cpu = Array::new(vec![batch_size, char_classes], 0.0);
    let mut l3_out_d_cpu = Array::new(vec![batch_size, char_classes], 0.0);
    let mut l1_w_cpu = Array::new(vec![1+char_classes+l1_size, 4*l1_size], 0.0);
    let mut argmax_out = Array::new(vec![batch_size], 0usize);

    let samples = 10000;
    let mut i = 0;
    for line in &lines {
        for (t, &(input, _, train_out)) in (1..line.len()).zip(steps.iter()) {
            input.write(graph, &char_map[&line[t-1]]);
            train_out.write(graph, &char_map[&line[t]]);
        }
        graph.forward();
        graph.backward();

        for (t, &(_, _, _)) in (1..line.len()).zip(steps.iter()) {
            print!("{}", line[t] as char);
        }
        println!("");

        for (_, &(_, l3_out, _)) in (1..line.len()).zip(steps.iter()) {
            l3_out.read(graph, &mut l3_out_cpu);
            //l1_w.read(graph, &mut l1_w_cpu);
            //graph.get_gradient(l3_out).read(graph, &mut l3_out_d_cpu);
            //graph.get_gradient(l1_w).read(graph, &mut l1_w_cpu);
            //graph.get_gradient(last_recur_in.2).read(graph, &mut l2_out_cpu);
            //println!("{:?}", l3_out_cpu);
            //println!("{:?}", l3_out_d_cpu);
            //println!("{:?}", l1_w_cpu);
            //println!("{:?}", l2_out_cpu);
            util::argmax_rows(&l3_out_cpu, &mut argmax_out);
            let next_char = rev_char_map[&argmax_out[&[0]]];
            print!("{}", next_char as char);
        }
        println!("");

        train::apply_gradients(graph);

        println!("{}", i);

        i += 1;
        if i > samples {
            break;
        }
    }

    /*let mut l3_out_cpu = Array::new(vec![batch_size, char_classes], 0.0);
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
    }*/
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
