use anyhow::Result;
use clap::{Parser, Subcommand};
use log::info;
use stg_mamba::{STGMambaConfig, Backend};

#[derive(Parser)]
#[command(name = "stg-mamba")]
#[command(about = "STG-Mamba for financial time-series prediction", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train the STG-Mamba model
    Train {
        /// Path to configuration file
        #[arg(short, long, default_value = "config.toml")]
        config: String,

        /// Dataset to use
        #[arg(short, long)]
        dataset: String,

        /// Number of epochs
        #[arg(short, long, default_value = "100")]
        epochs: usize,

        /// Batch size
        #[arg(short, long, default_value = "32")]
        batch_size: usize,

        /// Learning rate
        #[arg(short, long, default_value = "0.001")]
        learning_rate: f64,
    },

    /// Run inference with a trained model
    Predict {
        /// Path to model checkpoint
        #[arg(short, long)]
        model: String,

        /// Path to input data
        #[arg(short, long)]
        input: String,

        /// Path to output predictions
        #[arg(short, long)]
        output: String,
    },

    /// Evaluate model on test data
    Evaluate {
        /// Path to model checkpoint
        #[arg(short, long)]
        model: String,

        /// Path to test data
        #[arg(short, long)]
        data: String,
    },
}

fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            config,
            dataset,
            epochs,
            batch_size,
            learning_rate,
        } => {
            info!("Starting training...");
            info!("Config: {}", config);
            info!("Dataset: {}", dataset);
            info!("Epochs: {}, Batch size: {}, LR: {}", epochs, batch_size, learning_rate);

            // TODO: Implement training
            println!("Training functionality will be implemented");
        }

        Commands::Predict { model, input, output } => {
            info!("Running prediction...");
            info!("Model: {}", model);
            info!("Input: {}", input);
            info!("Output: {}", output);

            // TODO: Implement prediction
            println!("Prediction functionality will be implemented");
        }

        Commands::Evaluate { model, data } => {
            info!("Evaluating model...");
            info!("Model: {}", model);
            info!("Data: {}", data);

            // TODO: Implement evaluation
            println!("Evaluation functionality will be implemented");
        }
    }

    Ok(())
}
