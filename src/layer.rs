use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use std::num::NonZeroUsize;

// ---------------------------
pub struct Weights
{
    pub values : Array2<f64>
}

impl std::fmt::Debug for Weights 
{
    fn fmt(&self, form: &mut std::fmt::Formatter) -> std::fmt::Result 
    {
        let nrows = self.values.nrows();
        let ncols = self.values.ncols();

        write!(form, "Weights matrix {nrows}x{ncols}")
    }
}

pub struct Biases
{
    pub values : Array2<f64>
}

impl std::fmt::Debug for Biases 
{
    fn fmt(&self, form: &mut std::fmt::Formatter) -> std::fmt::Result 
    {
        let ncols = self.values.ncols();

        write!(form, "Biases {ncols}")
    }
}
// ---------------------------
pub struct Layer 
{
    pub weights: Weights,
    pub biases : Biases ,
}
// ---------------------------
impl Layer 
{
    /// Initializer for layer
    /// Args:
    /// input_size : Unsigned representing size of input layer
    /// output_size: Unsigned with size of output layer
    ///
    /// Usage:
    /// ```
    /// use nets::layer::Layer;
    /// use std::num::NonZeroUsize;
    ///
    /// let in_size  = NonZeroUsize::new(3).unwrap();
    /// let out_size = NonZeroUsize::new(2).unwrap();
    ///
    /// let layer = Layer::new(in_size, out_size);
    /// ```
    pub fn new(input_size: NonZeroUsize, output_size: NonZeroUsize) -> Self 
    {
        let limit        = (1.0 / input_size.get() as f64).sqrt();
        let distribution = Uniform::new(-limit, limit);

        let wgt_shape    = (input_size.get(), output_size.get());
        let weights      = Weights{ values : Array2::random(wgt_shape, distribution) };

        let out_shape    = (1, output_size.get());
        let biases       = Biases{ values : Array2::zeros(out_shape) };

        Layer { weights, biases }
    }

    /// Computes output of current layer
    /// Args:
    /// input: 2D array representing outputs from current layer
    ///
    /// Returns:
    /// 2D array with inputs for next layer
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64>
    {
        input.dot(&self.weights.values) + &self.biases.values
    }
}
// ---------------------------
