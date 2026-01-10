use serde::{Deserialize, Serialize};

/// Configuration for the STG-Mamba model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Number of nodes in the graph
    pub num_nodes: usize,

    /// Number of input features per node
    pub in_channels: usize,

    /// Number of output features (prediction horizon)
    pub out_channels: usize,

    /// Hidden dimension for Mamba layers
    pub hidden_dim: usize,

    /// Number of STG-Mamba blocks
    pub num_layers: usize,

    /// Input sequence length
    pub seq_len: usize,

    /// Prediction horizon
    pub pred_len: usize,

    /// Dropout rate
    pub dropout: f64,

    /// State space dimension for Mamba
    pub state_dim: usize,

    /// Expansion factor for Mamba
    pub expand_factor: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            num_nodes: 307,
            in_channels: 1,
            out_channels: 1,
            hidden_dim: 64,
            num_layers: 3,
            seq_len: 12,
            pred_len: 12,
            dropout: 0.1,
            state_dim: 16,
            expand_factor: 2,
        }
    }
}

impl ModelConfig {
    /// Create configuration for PEMS04 dataset
    pub fn pems04() -> Self {
        Self {
            num_nodes: 307,
            in_channels: 3,
            out_channels: 1,
            hidden_dim: 64,
            num_layers: 3,
            seq_len: 12,
            pred_len: 12,
            dropout: 0.1,
            state_dim: 16,
            expand_factor: 2,
        }
    }

    /// Create configuration for KnowAir dataset
    pub fn knowair() -> Self {
        Self {
            num_nodes: 184,
            in_channels: 1,
            out_channels: 1,
            hidden_dim: 64,
            num_layers: 3,
            seq_len: 24,
            pred_len: 24,
            dropout: 0.1,
            state_dim: 16,
            expand_factor: 2,
        }
    }

    /// Create configuration for HZ Metro dataset
    pub fn hz_metro() -> Self {
        Self {
            num_nodes: 80,
            in_channels: 1,
            out_channels: 1,
            hidden_dim: 64,
            num_layers: 3,
            seq_len: 12,
            pred_len: 12,
            dropout: 0.1,
            state_dim: 16,
            expand_factor: 2,
        }
    }

    /// Create configuration for financial data
    pub fn financial(num_assets: usize) -> Self {
        Self {
            num_nodes: num_assets,
            in_channels: 5, // OHLCV
            out_channels: 1, // predicted price/return
            hidden_dim: 128,
            num_layers: 4,
            seq_len: 60, // 60 time steps lookback
            pred_len: 20, // 20 steps ahead prediction
            dropout: 0.15,
            state_dim: 32,
            expand_factor: 2,
        }
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub learning_rate: f64,
    pub num_epochs: usize,
    pub weight_decay: f64,
    pub grad_clip: f64,
    pub patience: usize,
    pub checkpoint_dir: String,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            learning_rate: 0.001,
            num_epochs: 100,
            weight_decay: 0.0001,
            grad_clip: 5.0,
            patience: 10,
            checkpoint_dir: "checkpoints".to_string(),
        }
    }
}
