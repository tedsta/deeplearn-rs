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
    // Training data
    println!("Reading training labels...");
    let train_labels = read_mnist_labels("data/mnist/train-labels-idx1-ubyte", None).unwrap();
    let train_labels_logits: Vec<Array<f32>> = train_labels.iter().cloned()
                                                           .map(|x| util::one_hot_rows(x, 10))
                                                           .collect();
    println!("Label count: {}", train_labels.len());

    println!("Reading training images...");
    let (rows, columns, train_images) = read_mnist_images("data/mnist/train-images-idx3-ubyte", None).unwrap();

    // Validation data
    println!("Reading validation labels...");
    let val_labels = read_mnist_labels("data/mnist/t10k-labels-idx1-ubyte", Some(1000)).unwrap();
    println!("Label count: {}", val_labels.len());

    println!("Reading validation images...");
    let (_, _, val_images) = read_mnist_images("data/mnist/t10k-images-idx3-ubyte", Some(1000)).unwrap();

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Build the graph

    let ctx = Rc::new(ga::Context::new());
    let ref mut graph = Graph::new(ctx.clone());

    let batch_size = 1;

    //////////////////////////
    // Layer 1

    // Input. 1 batch of rows*columns inputs
    let input = graph.add_variable(vec![batch_size, rows*columns], false, vec![0.0; rows*columns]);

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
    // Train and validate the network

    // We apply a gradient of -0.001 to the loss function
    let loss_d_cpu = Array::new(vec![batch_size, 10], -0.001);
    loss_d.write(graph, &loss_d_cpu);

    let mut loss_out_cpu = Array::new(vec![batch_size, 10], 0.0);
    let mut l2_out_cpu = Array::new(vec![batch_size, 10], 0.0);
    let mut l2_out_d_cpu = Array::new(vec![batch_size, 10], 0.0);

    let mut predictions = Array::new(vec![batch_size], 0usize);
    let mut num_correct = 0;

    {
        // Put this in it's own scope so that our train_update closure doesn't hold onto all of our
        // stuff until the end of main()
        let train_update = |graph: &mut Graph, epoch: usize| {
            // Get the output
            l2_out.read(graph, &mut l2_out_cpu);

            // Get the most likely digit (the index of the neuron with the highest output)
            util::argmax_rows(&l2_out_cpu, &mut predictions);
            let prediction = predictions[&[0]];

            // Check if the model was correct
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
                      &[(input, &train_images), (train_out, &train_labels_logits)]);
    }

    /////////////////////////
    // Validate the network
    println!("#######################################");
    println!("Validating");
    num_correct = 0;
    for epoch in 0..val_images.len() {
        // Upload training data
        input.write(graph, &val_images[epoch]);

        // Run the graph
        graph.forward();

        // Get the output
        l2_out.read(graph, &mut l2_out_cpu);

        // Get the most likely digit (the index of the neuron with the highest output)
        util::argmax_rows(&l2_out_cpu, &mut predictions);
        let prediction = predictions[&[0]];

        // Check if the model was correct
        if prediction == val_labels[epoch] as usize {
            num_correct += 1;
        }
    }
    println!("Validation Accuracy: {}%", (num_correct as f32)/(val_images.len() as f32) * 100.0);
}

fn read_mnist_labels<P: AsRef<Path>>(path: P, num_samples: Option<usize>) -> io::Result<Vec<u8>> {
    use std::cmp;
    use std::io::{Error, ErrorKind};

    let ref mut file = BufReader::new(File::open(path).unwrap());

    let magic = u32::from_be(try!(read_u32(file)));
    if magic != 2049 {
        return Err(Error::new(ErrorKind::Other,
                              format!("Invalid magic number. Got expect 2049, got {}",
                                      magic).as_ref()))
    }

    let label_count = u32::from_be(try!(read_u32(file))) as usize;
    let label_count = cmp::min(label_count, num_samples.unwrap_or(label_count));

    let mut labels = Vec::with_capacity(label_count);
    for _ in 0..label_count {
        labels.push(try!(read_u8(file)));
    }

    Ok(labels)
}

fn read_mnist_images<P: AsRef<Path>>(path: P, num_samples: Option<usize>)
                                     -> io::Result<(usize, usize, Vec<Array<f32>>)> {
    use std::cmp;
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

    let image_count = cmp::min(image_count, num_samples.unwrap_or(image_count));

    let mut images = Vec::with_capacity(image_count);
    for _ in 0..image_count {
        let mut pixel_buf = vec![0u8; rows*columns];
        try!(file.read_exact(pixel_buf.as_mut()));
        let array = Array::from_vec(vec![rows, columns],
                                    pixel_buf.into_iter().map(|x| (x as f32)/255.0).collect());
        images.push(array);
    }

    Ok((rows, columns, images))
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
