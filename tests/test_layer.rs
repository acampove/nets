use nets::layer::Layer;

#[test]
fn new_creates_correct_shapes() 
{
    let layer = Layer::new(3, 2);

    assert_eq!(layer.weights.0.shape(), &[3, 2]);
    assert_eq!(layer.biases.0.shape(), &[1, 2]);
}

