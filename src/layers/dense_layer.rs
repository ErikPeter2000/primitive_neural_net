use ndarray::Array2;
use crate::activation_functions::activation_function::ActivationFunction;
use crate::layers::layer::Layer;


pub struct DenseLayer {
    weights: Array2<f64>,
    bias: Array2<f64>,
    activation: Box<dyn ActivationFunction>,
}

impl DenseLayer{
    pub fn new(weights: Array2<f64>, bias: Array2<f64>, activation: Box<dyn ActivationFunction>) -> DenseLayer {
        DenseLayer {
            weights,
            bias,
            activation,
        }
    }
}

impl Layer for DenseLayer {
    fn forward(&self, x: Array2<f64>) -> Array2<f64> {
        return self.activation.forward(x.dot(&self.weights) + &self.bias);
    }

    fn backward(&self, x: Array2<f64>) -> Array2<f64> {
        return self.activation.forward(x.dot(&self.weights.t()))
    }
}