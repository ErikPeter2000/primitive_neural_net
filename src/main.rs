mod activation_functions;
mod layers;
mod networks;
mod loss_functions;

use find_folder::Search;
use std::path::Path;
use crate::activation_functions::{Sigmoid, Softmax};
use crate::networks::Network;
use crate::loss_functions::{CrossEntropyLoss, MSE};
use csv;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array2, ArrayView2};
use plotters::prelude::*;
use mnist::{Mnist, MnistBuilder};

const LEARNING_RATE: f64 = 0.1;
const EPOCHS: usize = 10;
const TRAINING_SIZE: usize = 2000;
const TEST_SIZE: usize = 1000;

const INPUT_SIZE: usize = 784;
const OUTPUT_SIZE: usize = 10;

fn save_csv(data: &Vec<f64>, filename: &str, header: &str) {
    let mut wtr = csv::Writer::from_path(filename).unwrap();
    wtr.write_record(&[header]).unwrap();
    for value in data {
        wtr.write_record(&[value.to_string()]).unwrap();
    }
    wtr.flush().unwrap();
}

fn plot_loss(loss_history: &Vec<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("loss_plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Loss vs Epochs", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..loss_history.len(), 0.0..*loss_history.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        (0..).zip(loss_history.iter()).map(|(x, y)| (x, *y)),
        &RED,
    ))?;

    Ok(())
}

fn format_data(labels: &Vec<u8>, data_size: usize, scale: f64) -> Result<Array2<f64>, ndarray::ShapeError> {
    let labels: Vec<f64> = labels.iter().map(|&x| (x as f64) * scale).collect();
    Array2::from_shape_vec((labels.len() / data_size, data_size), labels)
}

fn format_data_one_hot(labels: &Vec<u8>, data_size: usize) -> Result<Array2<f64>, ndarray::ShapeError> {
    let length = labels.len();
    let labels: Vec<f64> = labels.iter()
        .flat_map(|&label| {
            let mut one_hot = vec![0.0; data_size];
            one_hot[label as usize] = 1.0;
            one_hot
        })
        .collect();
    Array2::from_shape_vec((length, data_size), labels)
}

fn main() {
    let mnist_path = Search::ParentsThenKids(3, 3)
        .for_folder("mnist")
        .expect("Could not find the mnist folder");
    let mnist_path = mnist_path.to_str().unwrap();
    println!("Mnist path: {}", mnist_path);

    let Mnist {
        trn_img, // Training images
        trn_lbl, // Training labels
        tst_img, // Test images
        tst_lbl, // Test labels
        ..
    } = MnistBuilder::new()
        .base_path(mnist_path)
        .label_format_digit() // Labels as digits (0-9)
        .training_set_length(TRAINING_SIZE as u32)
        .test_set_length(TEST_SIZE as u32)
        .finalize();

        let train_images = format_data(&trn_img, INPUT_SIZE, 1.0/255.0).expect("Error parsing training images");
    
        // Convert the training labels into a 2D array (60,000 x 1)
        let train_labels = format_data_one_hot(&trn_lbl, OUTPUT_SIZE).expect("Error parsing training images");

    // Create a new network
    let mut network = Network::new(Box::new(MSE));
    network.add_input_layer(784);
    network.add_dense_layer(128, Box::new(Sigmoid)).unwrap();
    network.add_dense_layer(128, Box::new(Sigmoid)).unwrap();
    network.add_dense_layer(10, Box::new(Softmax)).unwrap();

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
        for (input, target) in train_images.outer_iter().zip(train_labels.outer_iter()) {
            let input = ArrayView2::from_shape((INPUT_SIZE, 1), input.as_slice().unwrap()).unwrap();
            let target = ArrayView2::from_shape((OUTPUT_SIZE, 1), target.as_slice().unwrap()).unwrap();

            loss += network.training_pass(input.to_owned(), target.to_owned(), LEARNING_RATE);
        }
        loss_history.push(loss);
        bar.set_message(format!("Loss: {:.4}", loss));
        bar.inc(1);
    }
    bar.finish_with_message("Training complete");

    // Save the loss history to a CSV file
    println!("======== Summary ========");
    println!("Final Loss: {:?}", loss_history.last().unwrap());
    save_csv(&loss_history, "loss.csv", "loss");
    plot_loss(&loss_history).unwrap();

    // Test the network
    let test_images = format_data(&tst_img, INPUT_SIZE, 1.0/255.0).expect("Error converting parsing test images");
    let test_labels = format_data_one_hot(&tst_lbl, OUTPUT_SIZE).expect("Error converting parsing test labels");

    let mut correct = 0;
    for (input, target) in test_images.outer_iter().zip(test_labels.outer_iter()) {
        let input = ArrayView2::from_shape((INPUT_SIZE, 1), input.as_slice().unwrap()).unwrap();
        let target = ArrayView2::from_shape((OUTPUT_SIZE, 1), target.as_slice().unwrap()).unwrap();

        let output = network.forward_pass(input.to_owned());
        let prediction = output.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let target = target.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;

        if prediction == target {
            correct += 1;
        }
    }

    println!("Accuracy: {:.2}%", (correct as f64 / (TEST_SIZE as f64)) * 100.0);
}
