use nets::layer::Layer;
use std::num::NonZeroUsize;

#[test]
fn new_creates_correct_shapes() 
{
    let in_size  = NonZeroUsize::new(3).unwrap();
    let out_size = NonZeroUsize::new(2).unwrap();

    let layer = Layer::new(in_size, out_size);

    assert_eq!(layer.weights.0.shape(), &[3, 2]);
    assert_eq!(layer.biases.0.shape(), &[1, 2]);
}

