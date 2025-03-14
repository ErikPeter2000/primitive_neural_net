use ndarray::Array2;

pub trait Layer {
    fn get_neuron_count(&self) -> usize;
    fn forward(&mut self, x: Array2<f64>) -> Array2<f64>;
    fn backward(&mut self, chain: Array2<f64>, learning_rate: f64) -> Array2<f64>;
}
