use crate::activation_functions::ActivationFunction;
use ndarray::Array2;

pub struct Softmax;

impl ActivationFunction for Softmax {
    fn forward(&self, x: Array2<f64>) -> Array2<f64> {
        let exps = x.mapv(f64::exp);
        let sum = exps.sum();
        exps / sum
    }

    fn backward(&self, x: Array2<f64>) -> Array2<f64> {
        let exps = x.mapv(f64::exp);
        let sum = exps.sum();
        let softmax = exps / sum;
        softmax.to_owned() * (1.0 - softmax)
    }
}