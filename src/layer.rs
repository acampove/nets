use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

// ---------------------------
pub struct Weights(pub Array2<f64>);

impl std::fmt::Debug for Weights 
{
    fn fmt(&self, form: &mut std::fmt::Formatter) -> std::fmt::Result 
    {
        let nrows = self.0.nrows();
        let ncols = self.0.ncols();

        write!(form, "Weights matrix {nrows}x{ncols}")
    }
}

pub struct Biases(pub Array2<f64>);

impl std::fmt::Debug for Biases 
{
    fn fmt(&self, form: &mut std::fmt::Formatter) -> std::fmt::Result 
    {
        let ncols = self.0.ncols();

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
    ///
    /// let layer = Layer::new(3, 2);
    /// ```
    pub fn new(input_size: usize, output_size: usize) -> Self 
    {
        let limit        = (1.0 / input_size as f64).sqrt();
        let distribution = Uniform::new(-limit, limit);

        let wgt_shape    = (input_size, output_size);
        let weights      = Weights(Array2::random(wgt_shape, distribution));

        let out_shape    = (1, output_size);
        let biases       = Biases(Array2::zeros(out_shape));

        Layer { weights, biases }
    }
}
// ---------------------------
