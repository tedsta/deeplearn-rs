use matrix;
use matrix::ClMatrix;
use matrix::cl_matrix::ClMatrixMode;

use node::Node;

/// M = input count
/// N = node count
pub struct Activation {
    pub in_sum: ClMatrix<f32>,  // 1xN
    pub output: ClMatrix<f32>,  // 1xN
    pub ig: ClMatrix<f32>,      // 1xM : Input gradients
    pub wg: ClMatrix<f32>,      // MxN : Weight gradients
    pub weights: ClMatrix<f32>, // MxN
}

impl Activation {
    pub fn new(ctx: &matrix::Context, weights: ClMatrix<f32>) -> Activation {
        Activation {
            in_sum: ClMatrix::new(ctx, 1, weights.columns(), ClMatrixMode::Mut),
            output: ClMatrix::new(ctx, 1, weights.columns(), ClMatrixMode::Mut),
            ig: ClMatrix::new(ctx, weights.rows(), weights.columns(), ClMatrixMode::Mut),
            wg: ClMatrix::new(ctx, weights.rows(), weights.columns(), ClMatrixMode::Mut),
            weights: weights,
        }
    }
}

impl Node for Activation {
    /// Input: 1xM
    fn step(&self, ctx: &matrix::Context, input: &ClMatrix<f32>) {
        //let w_trans = ClMatrix::new(ctx, self.weights.columns(), self.weights.rows(), ClMatrixMode::Mut);
        //self.weights.transpose(ctx, &w_trans);
        input.multiply(ctx, &self.weights, &self.in_sum); // in_sum = input*weights
        self.in_sum.max(ctx, 0.0, &self.output);
    }

    /// gradients: NxP
    fn back_prop(&mut self, ctx: &matrix::Context, input: &ClMatrix<f32>, gradients: &ClMatrix<f32>) {
        /*for i in 0..input.len() {
            self.ig[i] = 0.0;
            self.wg[i] = 0.0;
        }
        for gradient in gradients.iter().cloned() {
            for i in 0..input.len() {
                self.ig[i] += dsigmoid(self.in_sum)*self.weights[i]*gradient;
                self.wg[i] += dsigmoid(self.in_sum)*input[i]*gradient;
            }
        }*/
    }
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
}

pub fn dsigmoid(x: f64) -> f64 {
    (1.0 - sigmoid(x)) * sigmoid(x)
}

#[test]
fn it_works() {
    let inputs = vec![0.5, 0.2];
    let mut n = Activation::new(vec![0.7, 0.1]);
    n.step(&inputs);
    let first_output = n.output;
    n.back_prop(&inputs, &vec![-1.0, 1.0]);
    for i in 0..n.weights.len() {
        n.weights[i] += n.wg[i]*0.1;
    }
    n.step(&inputs);
    let second_output = n.output;

    assert!(first_output == second_output);
}
