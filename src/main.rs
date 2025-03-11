mod activation_functions;
mod layers;

use ndarray::Array2;
use crate::layers::dense_layer::DenseLayer;
use crate::activation_functions::sigmoid::Sigmoid;

fn main() {
    let layer = DenseLayer::new(
        Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap(),
        Box::new(Sigmoid::new()),
    );
}
