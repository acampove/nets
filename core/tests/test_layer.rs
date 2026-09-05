use nets::layer::Layer;
use ndarray::Array2;
use std::num::NonZeroUsize;

#[test]
fn inizialize() 
{
    let in_size  = NonZeroUsize::new(3).unwrap();
    let out_size = NonZeroUsize::new(2).unwrap();

    let layer = Layer::new(in_size, out_size);

    assert_eq!(layer.weights.values.shape(), &[3, 2]);
    assert_eq!(layer.biases.values.shape(), &[1, 2]);
}

#[test]
fn forward() 
{
    let in_size  = NonZeroUsize::new(3).unwrap();
    let out_size = NonZeroUsize::new(2).unwrap();

    let layer  = Layer::new(in_size, out_size);
    let input  = Array2::<f64>::zeros((1, 3));

    let _values = layer.forward(&input);
}

