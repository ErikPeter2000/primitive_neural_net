use ndarray::Array2;

pub trait Layer {
    fn forward(&self, x: Array2<f64>) -> Array2<f64>;
    fn backward(&self, x: Array2<f64>) -> Array2<f64>;
}