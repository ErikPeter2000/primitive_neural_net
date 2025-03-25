use crate::activation_functions::ActivationFunction;
use crate::layers::Dense;
use crate::layers::Input;
use crate::layers::Layer;
use crate::loss_functions::LossFunction;

use ndarray::Array2;

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
    pub loss_function: Box<dyn LossFunction>,
}

impl Network {
    pub fn new(loss_function: Box<dyn LossFunction> ) -> Network {
        Network { layers: Vec::new(), loss_function }
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
        target: Array2<f64>,
        learning_rate: f64,
    ) {
        let mut chain = self.loss_function.gradient(output, target);
        for layer in self.layers.iter_mut().rev() {
            chain = layer.backward(chain, learning_rate);
        }
    }

    pub fn training_pass(&mut self, input: Array2<f64>, target: Array2<f64>, learning_rate: f64) -> f64 {
        let output = self.forward_pass(input.to_owned());
        self.backward_pass(output.clone(), target.clone(), learning_rate);
        let absolute_error = self.loss_function.loss(output.clone(), target.clone());
        return absolute_error.iter().fold(0.0, |acc, x| acc + x.powi(2));
    }
}
