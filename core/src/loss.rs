use ndarray::Array1;
use serde::Deserialize;

pub trait Loss {
    fn compute(&self, predictions : &Array1<f64>, targets : &Array1<f64>) -> f64;
    fn gradient(&self, predictions : &Array1<f64>, targets : &Array1<f64>) -> Array1<f64>;
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
    /// cross-entropy loss
    ///
    /// # Examples
    ///
    /// ```
    /// let toml_str = "epsilon = 1e-8";
    /// let cfg : nets::loss::CrossEntropyCfg = toml::from_str(toml_str).unwrap();
    /// let cet = nets::loss::CrossEntropy::new(cfg);
    /// ```
    pub fn new(cfg : CrossEntropyCfg) -> Self {
        Self { cfg }
    }
}

impl Loss for CrossEntropy {
    /// Computes loss given prediction and target
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::array;
    /// use nets::loss::{Loss, CrossEntropy, CrossEntropyCfg};
    ///
    /// let cfg = CrossEntropyCfg { epsilon: 1e-8 };
    /// let cet = CrossEntropy::new(cfg);
    ///
    /// let predictions = array![0.9, 0.1];
    /// let targets = array![1.0, 0.0];
    ///
    /// let loss = cet.compute(&predictions, &targets);
    /// assert!(loss > 0.0);
    /// ```
    fn compute(&self, predictions : &Array1<f64>, targets : &Array1<f64>) -> f64 {
        - predictions
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| t * (p.max(self.cfg.epsilon)).ln())
        .sum::<f64>()
    }

    /// Computes gradient for cross-entropy
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::array;
    /// use nets::loss::{Loss, CrossEntropy, CrossEntropyCfg};
    ///
    /// let cfg = CrossEntropyCfg { epsilon: 1e-8 };
    /// let cet = CrossEntropy::new(cfg);
    ///
    /// let predictions = array![1.0, 1.0];
    /// let targets = array![1.0, 2.0];
    ///
    /// let grad = cet.gradient(&predictions, &targets);
    /// let expected = array![-1.0, -2.0];
    /// assert!(grad == expected);
    /// ```
    fn gradient(&self, predictions : &Array1<f64>, targets : &Array1<f64>) -> Array1<f64> {
        predictions
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| - t / p)
        .collect::<Array1<f64>>()
    }
}
