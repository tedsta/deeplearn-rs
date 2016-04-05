# deeplearn-rs

Deep learning in Rust! This is my first shot at this. It's mostly just a proof of concept right now. The API will change.

### Status

Right now it is the bare minimum required to manually build and train simple networks that use matrix multiplication, addition, ReLU, and MSE loss. It's enough to train a primitive MNIST classifier, though. Check out the examples.

### Road map

- Allow datatypes other than `f32` and implement casting between arrays of primitive numeric types.
- Implement some automatic trainers such as SGD and Ada Grad.
- Provide utilities for working with data
    - images
    - tsv and csv
    - raw text data and word embeddings

### License

MIT
