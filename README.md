# deeplearn-rs

Deep learning in Rust! This is my first shot at this. It's mostly just a proof of concept right now. The API will change.

### Status

We have these models implemented (check out the examples folder):
- MNIST handwritten digit recognition
- char-rnn using LSTM

So far, we have the following layers implemented:

- Matrix multiply (fully connected)
- Add (for bias, for example)
- LSTM
- Softmax
- MSE loss
- Cross entropy loss

We have the following optimizers:
- SGD
- RMSProp

### Road map

- More layer types (in the order that I'll probably get to them)
    - Conv2d
    - Pooling
    - Dropout
- Allow datatypes other than `f32` and implement casting between arrays of primitive numeric types.
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
