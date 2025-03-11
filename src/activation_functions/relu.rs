use crate::activation_functions::activation_function::ActivationFunction;

pub struct ReLU;

impl ReLU {
    pub fn new() -> ReLU {
        ReLU {}
    }
}

impl ActivationFunction for ReLU {
    fn forward(&self, x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    }

    fn backward(&self, x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}