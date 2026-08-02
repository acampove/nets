use nets::network::Network;
use std::num::NonZeroUsize;

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


