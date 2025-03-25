use ndarray::Array2;

pub trait LossFunction {
    fn loss(&self, output: Array2<f64>, target: Array2<f64>) -> Array2<f64>;
    fn gradient(&self, output: Array2<f64>, target: Array2<f64>) -> Array2<f64>;
}