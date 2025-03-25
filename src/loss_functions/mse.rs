use crate::loss_functions::loss_function::LossFunction;
use ndarray::Array2;

pub struct MSE;

impl LossFunction for MSE {
    fn loss(&self, output: Array2<f64>, target: Array2<f64>) -> Array2<f64> {
        let diff = &output - &target;
        let diff_squared = diff.mapv(|x| x.powi(2));
        diff_squared * 0.5
    }

    fn gradient(&self, output: Array2<f64>, target: Array2<f64>) -> Array2<f64> {
        let diff = &output - &target;
        diff
    }
}