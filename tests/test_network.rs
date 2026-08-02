use nets::network::Network;
use std::num::NonZeroUsize;
use ndarray::array;

#[test]
fn inizialize() 
{
    let sizes = [
        NonZeroUsize::new(3).unwrap(),
        NonZeroUsize::new(4).unwrap(),
        NonZeroUsize::new(2).unwrap(),
    ];

    let _network = Network::new(&sizes);
}

#[test]
fn forward() 
{
    let sizes = [
        NonZeroUsize::new(3).unwrap(),
        NonZeroUsize::new(4).unwrap(),
        NonZeroUsize::new(2).unwrap(),
    ];

    let network = Network::new(&sizes);
    let input   = array![[1.0, 2.0, 3.0]];
    let output  = network.forward(&input);

    assert_eq!(output.shape(), &[1, 2]);
}
