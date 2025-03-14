use crate::activation_functions::ActivationFunction;
use crate::layers::Layer;
use ndarray::Array2;
use rand;

pub struct Dense {
    weights: Array2<f64>,
    bias: Array2<f64>,
    neuron_count: usize,
    activation: Box<dyn ActivationFunction>,
    layer_input: Array2<f64>,
    pre_activation: Array2<f64>,
    layer_output: Array2<f64>,
}

impl Dense {
    pub fn new(
        neurons: usize,
        previous_neurons: usize,
        activation: Box<dyn ActivationFunction>,
    ) -> Dense {
        Dense {
            weights: Array2::from_shape_fn((neurons, previous_neurons), |_| {
                rand::random::<f64>() * 2.0 - 1.0
            }),
            bias: Array2::from_shape_fn((neurons, 1), |_| rand::random::<f64>() * 2.0 - 1.0),
            neuron_count: neurons,
            activation,
            layer_input: Array2::zeros((1, 1)),
            pre_activation: Array2::zeros((1, 1)),
            layer_output: Array2::zeros((1, 1)),
        }
    }
}

impl Layer for Dense {
    fn forward(&mut self, x: Array2<f64>) -> Array2<f64> {
        self.layer_input = x.clone();
        self.pre_activation = self.weights.dot(&x) + &self.bias;
        self.layer_output = self.activation.forward(self.pre_activation.clone());
        return self.layer_output.clone();
    }

    fn backward(&mut self, chain: Array2<f64>, learning_rate: f64) -> Array2<f64> {
        let loss_vs_pre_activation =
            chain.clone() * self.activation.backward(self.pre_activation.clone());
        let weights_derivative = loss_vs_pre_activation.dot(&self.layer_input.t());
        let bias_derivative = loss_vs_pre_activation.clone();
        let chain = self.weights.t().dot(&loss_vs_pre_activation);

        self.weights = &self.weights - learning_rate * &weights_derivative;
        self.bias = &self.bias - learning_rate * &bias_derivative;

        chain
    }

    fn get_neuron_count(&self) -> usize {
        self.neuron_count
    }
}
