use ga;
use ndarray;
use rand::prelude::ThreadRng;

use crate::CpuArray;

pub trait Initializer {
    fn init(self, rng: &mut ThreadRng, shape: ndarray::IxDyn) -> CpuArray<f32>;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/*impl Initializer for f32 {
    fn init(self, _: &mut ThreadRng, shape: Vec<usize>) -> ga::Array<f32> {
        ga::Array::new(shape, self)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl Initializer for Vec<f32> {
    fn init(self, _: &mut ThreadRng, shape: Vec<usize>) -> ga::Array<f32> {
        ga::Array::from_vec(shape, self)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Uniform(min, max)
pub struct Uniform(pub f32, pub f32);

impl Initializer for Uniform {
    fn init(self, rng: &mut ThreadRng, shape: Vec<usize>) -> ga::Array<f32> {
        use rand::Rng;

        let Uniform(min, max) = self;
        let vec = (0..shape[0]*shape[1]).map(|_| rng.next_f32()*(max-min) + min).collect();
        ga::Array::from_vec(shape, vec)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Normal(mean, standard deviation)
pub struct Normal(pub f32, pub f32);

impl Initializer for Normal {
    fn init(self, rng: &mut ThreadRng, shape: Vec<usize>) -> ga::Array<f32> {
        use rand::distributions::Sample;

        let Normal(mean, std_dev) = self;
        let mut dist = rand::distributions::Normal::new(mean as f64, std_dev as f64);
        let vec = (0..shape[0]*shape[1]).map(|_| dist.sample(rng) as f32).collect();
        ga::Array::from_vec(shape, vec)
    }
}*/
