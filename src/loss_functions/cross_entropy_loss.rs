use crate::loss_functions::loss_function::LossFunction;
use ndarray::Array2;

pub struct CrossEntropyLoss;

impl LossFunction for CrossEntropyLoss {
    fn loss(&self, output: Array2<f64>, target: Array2<f64>) -> Array2<f64> {
        let epsilon = 1e-7;
        let output = output.mapv(|x| x.max(epsilon).min(1.0 - epsilon));
        let loss = -&target * &output.mapv(f64::ln);
        loss
    }

    fn gradient(&self, output: Array2<f64>, target: Array2<f64>) -> Array2<f64> {
        let epsilon = 1e-7; // Small value to prevent division by zero
        let output = output.mapv(|x| x.max(epsilon).min(1.0 - epsilon)); // Clamp output
        -(&target / &output)
    }
}