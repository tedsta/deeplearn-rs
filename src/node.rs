use matrix;
use matrix::cl_matrix::ClMatrix;

pub trait Node {
    fn step(&self, ctx: &matrix::Context, inputs: &ClMatrix<f32>);
    fn back_prop(&mut self, ctx: &matrix::Context, inputs: &ClMatrix<f32>, gradients: &ClMatrix<f32>);
}
