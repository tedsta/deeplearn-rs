#![feature(zero_one)]

extern crate deeplearn;
extern crate gpuarray as ga;

use std::fs::File;
use std::io::{
    self,
    BufReader,
    Read,
};
use std::num::{Zero, One};
use std::path::Path;
use std::rc::Rc;

use deeplearn::Graph;
use deeplearn::op::{Add, MatMul, Mse, Relu};
use deeplearn::init;
use ga::Array;

fn main() {
    // Training data
    println!("Reading training labels...");
    let train_labels = read_mnist_labels("data/mnist/train-labels-idx1-ubyte", None).unwrap();
    let train_labels_logits: Vec<Array<f32>> = train_labels.iter().cloned().map(|x| one_hot(x, 10)).collect();
    println!("Label count: {}", train_labels.len());

    println!("Reading training images...");
    let (rows, columns, train_images) = read_mnist_images("data/mnist/train-images-idx3-ubyte", None).unwrap();

    // Validation data
    println!("Reading validation labels...");
    let val_labels = read_mnist_labels("data/mnist/t10k-labels-idx1-ubyte", Some(1000)).unwrap();
    let val_labels_logits: Vec<Array<f32>> = val_labels.iter().cloned().map(|x| one_hot(x, 10)).collect();
    println!("Label count: {}", train_labels.len());

    println!("Reading validation images...");
    let (_, _, val_images) = read_mnist_images("data/mnist/t10k-images-idx3-ubyte", Some(1000)).unwrap();

    let ctx = Rc::new(ga::Context::new());

    // Setup the graph. It's going to have 2 inputs, 3 hidden nodes in it's 1 hidden layer, and 1
    // output node
    let mut graph = Graph::new(ctx.clone());

    //////////////////////////
    // Layer 1

    // Input.  1 batches of rows*columns inputs each.
    let input = graph.add_variable((1, rows*columns), vec![0.0; rows*columns]);
    // Weights for layer 1: [rows*columns inputs x 1000 nodes]
    let l1_w = graph.add_variable((rows*columns, 1000), init::Normal(0.001, 0.005));
    // Use matrix multiplication to do a fully connected layer
    let l1_mat_mul = graph.add_node(MatMul(input, l1_w));
    // Grab VarIndex for l1_mat_mul's output
    let l1_mat_mul_out = l1_mat_mul.get(&graph).outputs[0]; 
    // 1000 biases; one for each node in layer 1
    let l1_b = graph.add_variable((1, 1000), init::Normal(0.001, 0.005));
    // Here we add the biases to the matrix multiplication output
    let l1_biased = graph.add_node(Add(l1_mat_mul_out, l1_b, 0));
    // Grab VarIndex for l1_biased's output
    let l1_biased_out = l1_biased.get(&graph).outputs[0];
    // Run the biased input*weight sums through an ReLU activation
    let l1_relu = graph.add_node(Relu(l1_biased_out));
    let l1_relu_out = l1_relu.get(&graph).outputs[0];

    //////////////////////////
    // Layer 2

    // Weights for layer 2: [1000 inputs x 10 nodes]
    let l2_w = graph.add_variable((1000, 10), init::Normal(0.001, 0.005));
    // Fully connected layer 2. Use layer 1's output as layer 2's input
    let l2_mat_mul = graph.add_node(MatMul(l1_relu_out, l2_w));
    // Grab VarIndex for l2_mat_mul's output
    let l2_mat_mul_out = l2_mat_mul.get(&graph).outputs[0];
    // 10 bias for 1 output node
    let l2_b = graph.add_variable((1, 10), init::Normal(0.001, 0.005));
    // Here we add the bias to the matrix multiplication output
    let l2_biased = graph.add_node(Add(l2_mat_mul_out, l2_b, 0));
    // Grab VarIndex for l2_biased's output
    let l2_biased_out = l2_biased.get(&graph).outputs[0];
    // Run the biased input*weight sums through an ReLU activation
    let l2_relu = graph.add_node(Relu(l2_biased_out));
    // Grab VarIndex for l2_relu's output
    let l2_relu_out = l2_relu.get(&graph).outputs[0];

    //////////////////////////
    // Loss

    // Add a variable for the training outputs: [1 batches x 10 output]
    let train_out = graph.add_variable((1, 10), vec![0.0; 10]);
    // Use mean squared error loss function
    let loss = graph.add_node(Mse(l2_relu_out, train_out));
    // Grab the VarIndex for loss's output
    let loss_out = loss.get(&graph).outputs[0];
    // Create a gradient to apply to the loss function
    let loss_d = graph.add_gradient(loss, 0);

    ////////////////////////////////////////////////////////////////////////////////////////////////

    // We apply a gradient of -0.001 to the loss function
    let loss_d_cpu = Array::from_vec(vec![1, 10], vec![-0.001; 10]);
    loss_d.get(&graph).set(&ctx, &loss_d_cpu);

    /////////////////////////
    // Train the network
    let mut correct = 0;
    for epoch in 0..60000 {
        // Upload training data
        let train_sample = epoch%train_images.len();
        input.get(&graph).set(&ctx, &train_images[train_sample]);
        train_out.get(&graph).set(&ctx, &train_labels_logits[train_sample]);

        graph.run();
        let out = l2_relu_out.get(&graph).get(&ctx);
        let l1_w_d = graph.get_input_gradient(l1_w).unwrap().get(&graph);
        let l2_w_d = graph.get_input_gradient(l2_w).unwrap().get(&graph);
        let l1_b_d = graph.get_input_gradient(l1_b).unwrap().get(&graph);
        let l2_b_d = graph.get_input_gradient(l2_b).unwrap().get(&graph);
        ga::add(&ctx, &l1_w.get(&graph), -1, &l1_w_d, &l1_w.get(&graph));
        ga::add(&ctx, &l2_w.get(&graph), -1, &l2_w_d, &l2_w.get(&graph));
        ga::add(&ctx, &l1_b.get(&graph), -1, &l1_b_d, &l1_b.get(&graph));
        ga::add(&ctx, &l2_b.get(&graph), -1, &l2_b_d, &l2_b.get(&graph));

        let (mut max_index, mut max_value) = (0, out[&[0, 0]]);
        for (i, val) in out.buffer().iter().enumerate() {
            if *val > max_value {
                max_index = i;
                max_value = *val;
            }
        }

        if max_index == train_labels[train_sample] as usize {
            correct += 1;
        }

        if epoch % 1000 == 999 {
            let out_d = l2_relu.get(&graph)
                               .out_grad[0]
                               .gradient()
                               .get(&graph)
                               .get(&ctx);
            let l = loss_out.get(&graph).get(&ctx);
            println!("===================");
            println!("Epoch: {}", epoch);
            println!("out = {:?}", out);
            println!("out_d = {:?}", out_d);
            println!("loss = {:?}", l);
            println!("Accuracy: {}%", (correct as f32)/(1000 as f32) * 100.0);
            correct = 0;
        }
    }

    /////////////////////////
    // Validate the network
    println!("#######################################");
    println!("Validating");
    let mut correct = 0;
    for epoch in 0..1000 {
        // Upload training data
        let train_sample = epoch%val_images.len();
        input.get(&graph).set(&ctx, &val_images[train_sample]);
        train_out.get(&graph).set(&ctx, &val_labels_logits[train_sample]);

        graph.run();
        let out = l2_relu_out.get(&graph).get(&ctx);

        let (mut max_index, mut max_value) = (0, out[&[0, 0]]);
        for (i, val) in out.buffer().iter().enumerate() {
            if *val > max_value {
                max_index = i;
                max_value = *val;
            }
        }

        if max_index == val_labels[train_sample] as usize {
            correct += 1;
        }

        if epoch % 1000 == 999 {
            let out_d = l2_relu.get(&graph)
                               .out_grad[0]
                               .gradient()
                               .get(&graph)
                               .get(&ctx);
            let l = loss_out.get(&graph).get(&ctx);
            println!("===================");
            println!("Epoch: {}", epoch);
            println!("out = {:?}", out);
            println!("out_d = {:?}", out_d);
            println!("loss = {:?}", l);
            println!("Accuracy: {}%", (correct as f32)/(1000 as f32) * 100.0);
            correct = 0;
        }
    }
}

fn read_mnist_labels<P: AsRef<Path>>(path: P, num_samples: Option<usize>) -> io::Result<Vec<u8>> {
    use std::io::{Error, ErrorKind};

    let ref mut file = BufReader::new(File::open(path).unwrap());

    let magic = u32::from_be(try!(read_u32(file)));
    if magic != 2049 {
        return Err(Error::new(ErrorKind::Other,
                              format!("Invalid magic number. Got expect 2049, got {}",
                                      magic).as_ref()))
    }

    let label_count = u32::from_be(try!(read_u32(file)));

    let mut labels = Vec::with_capacity(label_count as usize);
    for i in 0..label_count {
        if let Some(num_samples) = num_samples {
            if i as usize > num_samples {
                break;
            }
        }
        labels.push(try!(read_u8(file)));
    }

    Ok(labels)
}

fn read_mnist_images<P: AsRef<Path>>(path: P, num_samples: Option<usize>) -> io::Result<(usize, usize, Vec<Array<f32>>)> {
    use std::io::{Error, ErrorKind};

    let ref mut file = BufReader::new(File::open(path).unwrap());

    let magic = u32::from_be(try!(read_u32(file)));
    if magic != 2051 {
        return Err(Error::new(ErrorKind::Other,
                              format!("Invalid magic number. Got expect 2051, got {}",
                                      magic).as_ref()))
    }

    let image_count = u32::from_be(try!(read_u32(file))) as usize;
    let rows = u32::from_be(try!(read_u32(file))) as usize;
    let columns = u32::from_be(try!(read_u32(file))) as usize;

    let mut images = Vec::with_capacity(image_count);
    for i in 0..image_count {
        if let Some(num_samples) = num_samples {
            if i as usize > num_samples {
                break;
            }
        }
        let mut pixel_buf = vec![0u8; rows*columns];
        try!(file.read_exact(pixel_buf.as_mut()));
        let array = Array::from_vec(vec![rows, columns],
                                    pixel_buf.into_iter().map(|x| (x as f32)/255.0).collect());
        images.push(array);
    }

    Ok((rows, columns, images))
}

fn one_hot<N, M>(label: N, classes: N) -> Array<M>
    where usize: From<N>,
          M:     ga::num::Num+Zero+One,
{
    let classes: usize = From::from(classes); // Cast class count to usize
    let label: usize = From::from(label); // Cast label to usize
    let mut buf: Vec<M> = vec![Zero::zero(); classes]; // Create array of zeroes
    buf[label] = One::one(); // Set the one-hot component
    Array::from_vec(vec![1, buf.len()], buf) // Construct the array
}

fn read_u8<T: Read>(reader: &mut T) -> io::Result<u8> {
    use std::mem;

    let mut buf: [u8; 1] = [0];
    reader.read_exact(&mut buf).map(|_| {
        let data: u8 = unsafe { mem::transmute(buf) };
        data
    })
}

fn read_u32<T: Read>(reader: &mut T) -> io::Result<u32> {
    use std::mem;

    let mut buf: [u8; 4] = [0, 0, 0, 0];
    reader.read_exact(&mut buf).map(|_| {
        let data: u32 = unsafe { mem::transmute(buf) };
        data
    })
}
