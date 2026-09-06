//! Module with activation layer structs

use::ndarray::Array2;

pub trait Activation {
    fn forward(&self, z: &Array2<f64>) -> Array2<f64>;
    fn gradient(&self, z: &Array2<f64>) -> Array2<f64>;
}

/// ---------------------------
/// SIGMOID
/// ---------------------------
pub struct Sigmoid;

impl Activation for Sigmoid {
    fn forward(&self, z: &Array2<f64>) -> Array2<f64> {
        z.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn gradient(&self, z: &Array2<f64>) -> Array2<f64> {
        let a = self.forward(z);
        &a * &(1.0 - &a)
    }
}
