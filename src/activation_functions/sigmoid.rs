use crate::activation_functions::activation_function::ActivationFunction;
use ndarray::Array2;

pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Sigmoid {
        Sigmoid {}
    }
}

impl ActivationFunction for Sigmoid {
    fn forward(&self, x: Array2<f64>) -> Array2<f64> {
        1.0 / (1.0 + (-x).exp())
    }

    fn backward(&self, x: Array2<f64>) -> Array2<f64> {
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        sigmoid.clone() * (1.0 - sigmoid)
    }
}
