use crate::layer::Layer;
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
}
