use crate::activation_functions::activation_function::ActivationFunction;
use ndarray::Array2;

pub struct ReLU;

impl ReLU {
    pub fn new() -> ReLU {
        ReLU {}
    }
}

impl ActivationFunction for ReLU {
    fn forward(&self, x: Array2<f64>) -> Array2<f64> {
        x.mapv(|x| if x > 0.0 { x } else { 0.0 })
    }

    fn backward(&self, x: Array2<f64>) -> Array2<f64> {
        x.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}
