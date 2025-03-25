pub mod loss_function;
pub mod mse;
pub mod cross_entropy_loss;

pub use loss_function::LossFunction;
pub use mse::MSE;
pub use cross_entropy_loss::CrossEntropyLoss;