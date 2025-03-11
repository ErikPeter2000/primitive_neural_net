use ndarray::Array2;
use crate::activation_functions::activation_function::ActivationFunction;

pub struct Linear;

impl Linear {
    pub fn new() -> Linear {
        Linear {}
    }
}

impl ActivationFunction for Linear {
    fn forward(&self, x: Array2<f64>) -> Array2<f64> {
        x
    }

    fn backward(&self, _x: Array2<f64>) -> Array2<f64> {
        Array2::ones(_x.raw_dim())
    }
}