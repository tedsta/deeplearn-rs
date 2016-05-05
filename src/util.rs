use std::num::{Zero, One};

use ga::{self, Array};

pub fn one_hot_rows<N, M>(label: N, classes: N) -> Array<M>
    where usize: From<N>,
          M:     ga::num::Num+Zero+One,
{
    let classes: usize = From::from(classes); // Cast class count to usize
    let label: usize = From::from(label); // Cast label to usize
    let mut buf: Vec<M> = vec![Zero::zero(); classes]; // Create array of zeroes
    buf[label] = One::one(); // Set the one-hot component
    Array::from_vec(vec![1, buf.len()], buf) // Construct the array
}

pub fn argmax_rows(a: &Array<f32>, out: &mut Array<usize>) {
    let (rows, columns) = (a.shape()[0], a.shape()[1]);
    for row in 0..rows {
        // TODO: I would do this:
        // let max_col = (0..a.columns()).max_by_key(|col| a[&[row, *col]]);
        // But f32 does not implement Ord :'(
        let (mut max_col, mut max_val) = (0, a[&[row, 0]]);
        for col in 1..columns {
            let val = a[&[row, col]];
            if val > max_val {
                max_col = col;
                max_val = val;
            }
        }
        out[&[row]] = max_col;
    }
}
