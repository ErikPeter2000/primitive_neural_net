use crate::activation_functions::ActivationFunction;
use crate::layers::Dense;
use crate::layers::Input;
use crate::layers::Layer;

use ndarray::Array2;

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn new() -> Network {
        Network { layers: Vec::new() }
    }

    pub fn add_input_layer(&mut self, neurons: usize) {
        self.layers.push(Box::new(Input::new(neurons)));
    }

    pub fn add_dense_layer(
        &mut self,
        neurons: usize,
        activation: Box<dyn ActivationFunction>,
    ) -> Result<(), &'static str> {
        let previous_neurons = match self.layers.last() {
            Some(layer) => layer.get_neuron_count(),
            None => {
                return Err("Network must have a previous layer before adding a dense layer. Try adding an input layer first.");
            }
        };
        self.layers
            .push(Box::new(Dense::new(neurons, previous_neurons, activation)));
        Ok(())
    }

    pub fn forward_pass(&mut self, input: Array2<f64>) -> Array2<f64> {
        let mut output = input.clone();
        for layer in self.layers.iter_mut() {
            output = layer.forward(output);
        }
        output
    }

    pub fn backward_pass(
        &mut self,
        output: Array2<f64>,
        expected: Array2<f64>,
        learning_rate: f64,
    ) {
        let mut chain = output - expected; // MSE is hard-coded here. This should be generalised.
        for layer in self.layers.iter_mut().rev() {
            chain = layer.backward(chain, learning_rate);
        }
    }
}
