mod activation_functions;
mod layers;
mod networks;

use crate::activation_functions::Sigmoid;
use crate::networks::Network;
use csv;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array2;

const LEARNING_RATE: f64 = 5.0;
const EPOCHS: usize = 1000;
const TRAINING_DATA: [([f64; 2], [f64; 1]); 4] = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
];

fn save_csv(data: Vec<f64>, filename: &str, header: &str) {
    let mut wtr = csv::Writer::from_path(filename).unwrap();
    wtr.write_record(&[header]).unwrap();
    for value in data {
        wtr.write_record(&[value.to_string()]).unwrap();
    }
    wtr.flush().unwrap();
}

fn main() {
    // Create a new network
    let mut network = Network::new();
    network.add_input_layer(2);
    network.add_dense_layer(4, Box::new(Sigmoid)).unwrap();
    network.add_dense_layer(1, Box::new(Sigmoid)).unwrap();

    // Loss is hard-coded MSE. This should be generalised.
    let mut loss_history = Vec::<f64>::new();

    // Create a progress bar
    let bar = ProgressBar::new(EPOCHS as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] |{bar:40.cyan}| {pos}/{len} ({eta}) - {msg}",
            )
            .unwrap(),
    );

    // Train the network
    for _ in 0..EPOCHS {
        let mut loss = 0.0;
        for (input, target) in TRAINING_DATA.iter() {
            let input = Array2::from_shape_vec((2, 1), input.to_vec()).unwrap();
            let expected = Array2::from_shape_vec((1, 1), target.to_vec()).unwrap();

            let output = network.forward_pass(input.clone());
            let absolute_error = output.clone() - expected.clone();
            loss += absolute_error.iter().fold(0.0, |acc, x| acc + x.powi(2));

            network.backward_pass(output, expected, LEARNING_RATE);
        }
        loss_history.push(loss);
        bar.set_message(format!("Loss: {:.4}", loss));
        bar.inc(1);
    }
    bar.finish_with_message("Training complete");

    // Save the loss history to a CSV file
    println!("======== Summary ========");
    println!("Final Loss: {:?}", loss_history.last().unwrap());
    save_csv(loss_history, "loss.csv", "loss");
}
