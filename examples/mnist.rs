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

use deeplearn::{Graph, Trainer};
use deeplearn::op::Relu;
use deeplearn::init;
use deeplearn::layers;
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
    let ref mut graph = Graph::new(ctx.clone());

    //////////////////////////
    // Layer 1

    // Input. 1 batch of rows*columns inputs
    let input = graph.add_variable(vec![1, rows*columns], false, vec![0.0; rows*columns]);

    // Biased fully connected layer with 300 neurons
    let (l1_fcb, _, _) = layers::dense_biased(graph, input, 300,
                                              init::Normal(0.001, 0.005),  // Weights initializer
                                              init::Normal(0.001, 0.005)); // Bias initializer
    let l1_out = layers::activation(graph, Relu(l1_fcb));

    //////////////////////////
    // Layer 2

    // Biased fully connected layer with 10 neurons
    let (l2_fcb, _, _) = layers::dense_biased(graph, l1_out, 10,
                                              init::Normal(0.001, 0.005),  // Weights initializer
                                              init::Normal(0.001, 0.005)); // Bias initializer
    let l2_out = layers::activation(graph, Relu(l2_fcb));
    let l2_out_d = graph.get_gradient(l2_out);

    //////////////////////////
    // Loss

    let (loss_out, train_out) = layers::mse(graph, l2_out);
    let loss_d = graph.add_gradient(loss_out); // Create a gradient to apply to the loss function

    ////////////////////////////////////////////////////////////////////////////////////////////////

    // We apply a gradient of -0.001 to the loss function
    let loss_d_cpu = Array::new(vec![1, 10], -0.001);
    loss_d.get(graph).set(&ctx, &loss_d_cpu);

    let mut loss_out_cpu = Array::new(vec![1, 10], 0.0);
    let mut l2_out_cpu = Array::new(vec![1, 10], 0.0);
    let mut l2_out_d_cpu = Array::new(vec![1, 10], 0.0);

    let mut num_correct = 0;
    let train_update = |graph: &mut Graph, epoch: usize| {
        l2_out.read(graph, &mut l2_out_cpu);

        // I would do this:
        //
        // let prediction = (0..10).max_by_key(|i| l2_out_cpu[&[0, *i]]);
        //
        // But f32 does not implement Ord :'(
        let (mut prediction, mut pred_weight) = (0, l2_out_cpu[&[0, 0]]);
        for col in 0..10 {
            let val = l2_out_cpu[&[0, col]];
            if val > pred_weight {
                prediction = col;
                pred_weight = val;
            }
        }

        if prediction == train_labels[epoch] as usize {
            num_correct += 1;
        }

        if epoch % 1000 == 999 {
            l2_out_d.read(graph, &mut l2_out_d_cpu);
            loss_out.read(graph, &mut loss_out_cpu);
            println!("===================");
            println!("Epoch: {}", epoch);
            println!("out = {:?}", l2_out_cpu);
            println!("out_d = {:?}", l2_out_d_cpu);
            println!("loss = {:?}", loss_out_cpu);
            println!("Accuracy: {}%", (num_correct as f32)/(1000 as f32) * 100.0);
            num_correct = 0;
        }
    };

    let trainer = Trainer;
    trainer.train(graph, 60000, train_update,
                  &[l2_out],
                  &[(input, &train_images), (train_out, &train_labels_logits)]);

    /////////////////////////
    // Validate the network
    println!("#######################################");
    println!("Validating");
    let mut num_correct = 0;
    for epoch in 0..1000 {
        // Upload training data
        let train_sample = epoch%val_images.len();
        input.get(graph).set(&ctx, &val_images[train_sample]);
        train_out.get(graph).set(&ctx, &val_labels_logits[train_sample]);

        graph.forward();
        let out = l2_out.get(graph).get(&ctx);

        let (mut prediction, mut pred_weight) = (0, out[&[0, 0]]);
        for col in 0..10 {
            let val = out[&[0, col]];
            if val > pred_weight {
                prediction = col;
                pred_weight = val;
            }
        }

        if prediction == val_labels[train_sample] as usize {
            num_correct += 1;
        }

        if epoch % 1000 == 999 {
            println!("===================");
            println!("Epoch: {}", epoch);
            println!("out = {:?}", out);
            println!("Accuracy: {}%", (num_correct as f32)/(1000 as f32) * 100.0);
            num_correct = 0;
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

fn read_mnist_images<P: AsRef<Path>>(path: P, num_samples: Option<usize>)
                                     -> io::Result<(usize, usize, Vec<Array<f32>>)> {
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
