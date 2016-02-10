#![feature(zero_one)]

extern crate deeplearn;
extern crate matrix;

use std::fs::File;
use std::io::{
    self,
    BufReader,
    Read,
};
use std::num::{Zero, One};
use std::path::Path;

use deeplearn::Graph;
use deeplearn::op::{Add, MatMul, Mse, Relu};
use deeplearn::init;
use matrix::Matrix;

fn main() {
    println!("Reading training labels...");
    let train_labels = read_mnist_labels("data/mnist/train-labels-idx1-ubyte");
    let train_labels: Vec<Matrix<f32>> = train_labels.unwrap().into_iter().map(|x| one_hot(x, 10)).collect();
    println!("Label count: {}", train_labels.len());

    println!("Reading training images...");
    let (rows, columns, train_images) = read_mnist_images("data/mnist/train-images-idx3-ubyte").unwrap();

    let ctx = matrix::Context::new();

    // Setup the graph. It's going to have 2 inputs, 3 hidden nodes in it's 1 hidden layer, and 1
    // output node
    let mut graph = Graph::new();

    //////////////////////////
    // Layer 1

    // Input.  1 batches of rows*columns inputs each.
    let input = graph.add_variable(&ctx, (1, rows*columns), vec![0.0; rows*columns]);
    // Weights for layer 1: [2 inputs x 3 nodes]
    let l1_w = graph.add_variable(&ctx, (rows*columns, 1000), init::Normal(0.001, 0.005));
    // Use matrix multiplication to do a fully connected layer
    let l1_mat_mul = graph.add_node(&ctx,
                                    MatMul::new(&ctx, (1, rows*columns), (rows*columns, 1000)),
                                    vec![input, l1_w],
                                    &[(1, 1000)]); // out shape: [5x2]*[2x3] = [5 batches x 3 outputs]
    // Grab VarIndex for l1_mat_mul's output
    let l1_mat_mul_out = l1_mat_mul.get(&graph).outputs[0]; 
    // 3 biases; one for each node in layer 1
    let l1_b = graph.add_variable(&ctx, (1, 1000), init::Normal(0.001, 0.005));
    // Here we add the biases to the matrix multiplication output
    let l1_biased = graph.add_node(&ctx,
                               Add::new(0),
                               vec![l1_mat_mul_out, l1_b],
                               &[(1, 1000)]);
    // Grab VarIndex for l1_biased's output
    let l1_biased_out = l1_biased.get(&graph).outputs[0];
    // Run the biased input*weight sums through an ReLU activation
    let l1_relu = graph.add_node(&ctx,
                                 Relu::new(),
                                 vec![l1_biased_out],
                                 &[(1, 1000)]);
    let l1_relu_out = l1_relu.get(&graph).outputs[0];

    //////////////////////////
    // Layer 2

    // Weights for layer 2: [3 inputs x 1 nodes]
    let l2_w = graph.add_variable(&ctx, (1000, 10), init::Normal(0.001, 0.005));
    // Fully connected layer 2. Use layer 1's output as layer 2's input
    let l2_mat_mul = graph.add_node(&ctx,
                                    MatMul::new(&ctx, (1, 1000), (1000, 10)),
                                    vec![l1_relu_out, l2_w],
                                    &[(1, 10)]); // out shape: [5x3]*[3x1] = [5 batches x 1 output]
    // Grab VarIndex for l2_mat_mul's output
    let l2_mat_mul_out = l2_mat_mul.get(&graph).outputs[0];
    // 1 bias for 1 output node
    let l2_b = graph.add_variable(&ctx, (1, 10), init::Normal(0.001, 0.005));
    // Here we add the bias to the matrix multiplication output
    let l2_biased = graph.add_node(&ctx,
                               Add::new(0),
                               vec![l2_mat_mul_out, l2_b],
                               &[(1, 10)]);
    // Grab VarIndex for l2_biased's output
    let l2_biased_out = l2_biased.get(&graph).outputs[0];
    // Run the biased input*weight sums through an ReLU activation
    let l2_relu = graph.add_node(&ctx,
                                 Relu::new(),
                                 vec![l2_biased_out],
                                 &[(1, 10)]);
    // Grab VarIndex for l2_relu's output
    let l2_relu_out = l2_relu.get(&graph).outputs[0];

    //////////////////////////
    // Loss

    // Add a variable for the training outputs: [5 batches x 1 output]
    let train_out = graph.add_variable(&ctx, (1, 10), vec![0.0; 10]);
    // Use mean squared error loss function
    let loss = graph.add_node(&ctx,
                              Mse::new(),
                              vec![l2_relu_out, train_out],
                              &[(1, 10)]);
    // Grab the VarIndex for loss's output
    let loss_out = loss.get(&graph).outputs[0];
    // Create a gradient to apply to the loss function
    let loss_d = graph.add_gradient(&ctx, loss, 0);

    ////////////////////////////////////////////////////////////////////////////////////////////////

    // We apply a gradient of -0.1 to the loss function
    let loss_d_cpu = Matrix::from_vec(1, 10, vec![-0.001; 10]);
    loss_d.get(&graph).set(&ctx, &loss_d_cpu);

    /////////////////////////
    // Run/Train the network
    for epoch in 0..100000 {
        // Upload training data
        let train_sample = epoch%train_images.len();
        input.get(&graph).set(&ctx, &train_images[train_sample]);
        train_out.get(&graph).set(&ctx, &train_labels[train_sample]);

        graph.run(&ctx);
        let out = l2_relu_out.get(&graph).get(&ctx);
        let out_d = l2_relu.get(&graph)
                           .out_grad[0]
                           .gradient()
                           .get(&graph)
                           .get(&ctx);
        let l = loss_out.get(&graph).get(&ctx);
        let l1_w_d = graph.get_input_gradient(l1_w).unwrap().get(&graph);
        let l2_w_d = graph.get_input_gradient(l2_w).unwrap().get(&graph);
        let l1_b_d = graph.get_input_gradient(l1_b).unwrap().get(&graph);
        let l2_b_d = graph.get_input_gradient(l2_b).unwrap().get(&graph);
        l1_w.get(&graph).add(&ctx, -1, &*l1_w_d, &*l1_w.get(&graph));
        l2_w.get(&graph).add(&ctx, -1, &*l2_w_d, &*l2_w.get(&graph));
        l1_b.get(&graph).add(&ctx, -1, &*l1_b_d, &*l1_b.get(&graph));
        l2_b.get(&graph).add(&ctx, -1, &*l2_b_d, &*l2_b.get(&graph));
        if epoch % 1000 == 1 {
            println!("===================");
            println!("Epoch: {}", epoch);
            println!("out = {:?}", out);
            println!("out_d = {:?}", out_d);
            println!("loss = {:?}", l);
        }
    }
}

fn read_mnist_labels<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
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
    for _ in 0..label_count {
        labels.push(try!(read_u8(file)));
    }

    Ok(labels)
}

fn read_mnist_images<P: AsRef<Path>>(path: P) -> io::Result<(usize, usize, Vec<Matrix<f32>>)> {
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
    for _ in 0..image_count {
        let mut pixel_buf = vec![0u8; rows*columns];
        try!(file.read_exact(pixel_buf.as_mut()));
        let matrix = Matrix::from_vec(rows, columns,
                                      pixel_buf.into_iter().map(|x| (x as f32)/255.0).collect());
        images.push(matrix);
    }

    Ok((rows, columns, images))
}

fn one_hot<N, M>(label: N, classes: N) -> Matrix<M>
    where usize: From<N>,
          M:     matrix::num::Num+Zero+One,
{
    let classes: usize = From::from(classes); // Cast class count to usize
    let label: usize = From::from(label); // Cast label to usize
    let mut buf: Vec<M> = vec![Zero::zero(); classes]; // Create array of zeroes
    buf[label] = One::one(); // Set the one-hot component
    Matrix::from_vec(1, buf.len(), buf) // Construct the matrix
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
