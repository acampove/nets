use ndarray::Array2;

/// Sigmoid activation function
pub fn sigmoid(x: &Array2<f64>) -> Array2<f64> 
{
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

