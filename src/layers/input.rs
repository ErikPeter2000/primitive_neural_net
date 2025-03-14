use crate::layers::Layer;
use ndarray::Array2;

pub struct Input {
    neuron_count: usize,
}

impl Input {
    pub fn new(neuron_count: usize) -> Input {
        Input { neuron_count }
    }
}

impl Layer for Input {
    fn forward(&mut self, x: Array2<f64>) -> Array2<f64> {
        x.clone()
    }

    fn backward(&mut self, chain: Array2<f64>, _: f64) -> Array2<f64> {
        chain.clone()
    }

    fn get_neuron_count(&self) -> usize {
        self.neuron_count
    }
}
