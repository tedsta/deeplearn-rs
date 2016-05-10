# deeplearn-rs

Deep learning in Rust! This is my first shot at this. It's mostly just a proof of concept right now. The API will change.

### Status

Right now it is the bare minimum required to build and train simple networks that use matrix multiplication, addition, ReLU, and MSE loss. It's enough to train a primitive MNIST classifier, though! Check out the examples.

### Road map

- Cooler layer types (in the order that I'll probably get to them)
    - LSTM (There's decent progress on this one)
    - Conv2d
    - Pooling
    - Dropout
- Allow datatypes other than `f32` and implement casting between arrays of primitive numeric types.
- Implement some special trainers like Ada Grad.
- Provide utilities for working with data
    - images
    - tsv and csv
    - raw text data and word embeddings

### Goals

We have a looong way to go :)

- Fast
- Easy to use
- Portable
- More control when you need it
- Easy to define custom layers
- Readable internal codebase

### License

MIT
