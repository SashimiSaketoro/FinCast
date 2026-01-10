//! # STG-Mamba: Spatial-Temporal Graph Learning via Selective State Space Model
//!
//! A Rust implementation of STG-Mamba using the Burn deep learning framework
//! for financial time-series prediction.
//!
//! ## Architecture Overview
//!
//! STG-Mamba combines:
//! - **Spatial processing**: Graph Neural Networks with dynamic adjacency (KFGN)
//! - **Temporal processing**: Mamba (Selective State Space Model) for sequence modeling
//! - **Multi-granularity**: Kalman filtering for temporal granularity integration
//!
//! ## Key Components
//!
//! - `models`: Core STG-Mamba architecture (STGMamba, GS3B)
//! - `layers`: Neural network layers (Mamba, GraphConv, KFGN, etc.)
//! - `config`: Configuration management
//! - `utils`: Utility functions for normalization and metrics
//!
//! ## References
//!
//! - Paper: https://arxiv.org/abs/2403.12418
//! - GitHub: https://github.com/LincanLi98/STG-Mamba

pub mod config;
pub mod data;
pub mod layers;
pub mod models;
pub mod utils;

// Re-export key types
pub use config::{ModelConfig, TrainingConfig};
pub use data::{Asset, AssetFeatures, KrakenClient, PolymarketClient};
pub use models::{STGMamba, STGMambaConfig};

use burn::backend::{Autodiff, NdArray};

/// Type alias for the backend used in training
pub type Backend = Autodiff<NdArray>;

/// Type alias for the backend used in inference
pub type InferenceBackend = NdArray;
