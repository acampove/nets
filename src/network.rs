use crate::layer::Layer;
use ndarray::Array2;
use std::num::NonZeroUsize;

/// Neural network struct, meant to be a collection of layers
pub struct Network
{
    /// Layers making up network
    pub layers: Vec<Layer>,
}

impl Network
{
    /// Network initializer
    ///
    /// Usage:
    /// ```
    /// use nets::network::Network;
    /// use std::num::NonZeroUsize;
    ///
    /// let sizes = [
    /// NonZeroUsize::new(3).unwrap(),
    /// NonZeroUsize::new(4).unwrap(),
    /// NonZeroUsize::new(2).unwrap(),
    /// ];
    ///
    /// let _network = Network::new(&sizes);
    /// ```
    pub fn new(sizes: &[NonZeroUsize]) -> Self
    {
        let layers = sizes
            .windows(2)
            .map(|pair| Layer::new(pair[0], pair[1]))
            .collect();

        Network { layers }
    }
    /// Runs input through every layer in sequence, returning the final output.
    ///
    /// # Examples
    ///
    /// ```
    /// use nets::network::Network;
    /// use std::num::NonZeroUsize;
    /// use ndarray::array;
    ///
    /// let sizes = [
    ///     NonZeroUsize::new(3).unwrap(),
    ///     NonZeroUsize::new(4).unwrap(),
    ///     NonZeroUsize::new(2).unwrap(),
    /// ];
    ///
    /// let network = Network::new(&sizes);
    /// let input   = array![[1.0, 2.0, 3.0]];
    /// let output  = network.forward(&input);
    ///
    /// assert_eq!(output.shape(), &[1, 2]);
    /// ```
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64>
    {
        let mut output = input.clone();

        for layer in &self.layers 
        {
            output = layer.forward(&output);
        }

        output
    }
}
