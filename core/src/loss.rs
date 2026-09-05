use ndarray::Array1;
use serde::Deserialize;

pub trait Loss {
    fn compute(&self, predictions : &Array1<f64>, targets : &Array1<f64>) -> f64;
}
/// Configuration for cross-entropy loss
///
/// # Examples
///
/// ```
/// let toml_str = "epsilon = 1e-8";
/// let cfg: nets::loss::CrossEntropyCfg = toml::from_str(toml_str).unwrap();
/// assert_eq!(cfg.epsilon, 1e-8);
/// ```
#[derive(Deserialize)]
pub struct CrossEntropyCfg {
    pub epsilon : f64,
}

pub struct CrossEntropy {
    cfg : CrossEntropyCfg,
}

impl CrossEntropy {
    pub fn new(cfg : CrossEntropyCfg) -> Self {
        Self { cfg }
    }
}

impl Loss for CrossEntropy {
    fn compute(&self, predictions : &Array1<f64>, targets : &Array1<f64>) -> f64 {
        - predictions
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| t * (p.max(self.cfg.epsilon)).ln())
        .sum::<f64>()
    }
}
