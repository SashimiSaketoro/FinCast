// STG-Mamba: Spatial-Temporal Graph Learning via Selective State Space Model
//
// Paper: https://arxiv.org/abs/2403.12418
// GitHub: https://github.com/LincanLi98/STG-Mamba
//
// Architecture Overview:
//
// STG-Mamba consists of stacked Graph Selective State Space Blocks (GS3B).
// Each GS3B contains:
// 1. KFGN (Kalman Filtering GNN): Processes spatial relationships with dynamic graph structures
// 2. ST-S3M (Mamba SSM): Captures temporal dependencies with selective state spaces
// 3. Feed-forward network: Additional transformation capacity
//
// The model is designed for spatial-temporal forecasting tasks, particularly
// financial time-series prediction across multiple assets/entities.

use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, LayerNorm, LayerNormConfig, Embedding, EmbeddingConfig},
    tensor::{backend::Backend, Tensor},
};

use super::gs3b::{GS3B, GS3BConfig};

/// Configuration for STG-Mamba model
#[derive(Config, Debug)]
pub struct STGMambaConfig {
    /// Number of nodes in the graph (e.g., number of assets)
    pub num_nodes: usize,

    /// Number of input features per node
    pub in_channels: usize,

    /// Number of output features (prediction targets)
    pub out_channels: usize,

    /// Hidden dimension for all layers
    pub hidden_dim: usize,

    /// Number of GS3B blocks
    pub num_layers: usize,

    /// State space dimension for Mamba
    pub state_dim: usize,

    /// Number of temporal granularities in KFGN
    #[config(default = 3)]
    pub num_granularities: usize,

    /// Expansion factor for Mamba
    #[config(default = 2)]
    pub expand_factor: usize,

    /// Convolution kernel size for Mamba
    #[config(default = 4)]
    pub d_conv: usize,

    /// Dropout rate
    #[config(default = 0.1)]
    pub dropout: f64,

    /// Feed-forward expansion factor
    #[config(default = 4)]
    pub ffn_expand: usize,

    /// Input sequence length
    pub seq_len: usize,

    /// Prediction horizon
    pub pred_len: usize,
}

impl STGMambaConfig {
    /// Create configuration for financial forecasting
    pub fn financial(num_assets: usize) -> Self {
        Self {
            num_nodes: num_assets,
            in_channels: 5, // OHLCV
            out_channels: 1, // Predicted return/price
            hidden_dim: 128,
            num_layers: 4,
            state_dim: 32,
            num_granularities: 3,
            expand_factor: 2,
            d_conv: 4,
            dropout: 0.15,
            ffn_expand: 4,
            seq_len: 60,
            pred_len: 20,
        }
    }

    /// Create configuration for PEMS04 traffic dataset
    pub fn pems04() -> Self {
        Self {
            num_nodes: 307,
            in_channels: 3,
            out_channels: 1,
            hidden_dim: 64,
            num_layers: 3,
            state_dim: 16,
            num_granularities: 3,
            expand_factor: 2,
            d_conv: 4,
            dropout: 0.1,
            ffn_expand: 4,
            seq_len: 12,
            pred_len: 12,
        }
    }

    /// Create configuration for KnowAir pollution dataset
    pub fn knowair() -> Self {
        Self {
            num_nodes: 184,
            in_channels: 1,
            out_channels: 1,
            hidden_dim: 64,
            num_layers: 3,
            state_dim: 16,
            num_granularities: 3,
            expand_factor: 2,
            d_conv: 4,
            dropout: 0.1,
            ffn_expand: 4,
            seq_len: 24,
            pred_len: 24,
        }
    }
}

/// STG-Mamba Model
///
/// Main model for spatial-temporal graph forecasting
#[derive(Module, Debug)]
pub struct STGMamba<B: Backend> {
    /// Store config values needed for forward pass
    hidden_dim: usize,
    pred_len: usize,
    out_channels: usize,
    num_layers: usize,
    in_channels: usize,

    /// Input embedding/projection
    input_proj: Linear<B>,

    /// Stacked GS3B blocks
    gs3b_blocks: Vec<GS3B<B>>,

    /// Output projection
    output_proj: Linear<B>,

    /// Final layer norm
    output_norm: LayerNorm<B>,

    /// Temporal positional embeddings
    temporal_embedding: Embedding<B>,
}

impl<B: Backend> STGMamba<B> {
    /// Create a new STG-Mamba model
    pub fn new(config: &STGMambaConfig, device: &B::Device) -> Self {
        // Input projection: [in_channels -> hidden_dim]
        let input_proj = LinearConfig::new(config.in_channels, config.hidden_dim)
            .with_bias(true)
            .init(device);

        // Create stacked GS3B blocks
        let mut gs3b_blocks = Vec::new();
        for _ in 0..config.num_layers {
            let gs3b_config = GS3BConfig {
                num_nodes: config.num_nodes,
                hidden_dim: config.hidden_dim,
                state_dim: config.state_dim,
                num_granularities: config.num_granularities,
                expand_factor: config.expand_factor,
                d_conv: config.d_conv,
                dropout: config.dropout,
                ffn_expand: config.ffn_expand,
            };
            gs3b_blocks.push(GS3B::new(&gs3b_config, device));
        }

        // Output projection: [hidden_dim -> pred_len * out_channels]
        let output_proj = LinearConfig::new(
            config.hidden_dim,
            config.pred_len * config.out_channels,
        )
        .with_bias(true)
        .init(device);

        // Output normalization
        let output_norm = LayerNormConfig::new(config.hidden_dim).init(device);

        // Temporal positional embeddings
        let temporal_embedding = EmbeddingConfig::new(config.seq_len, config.hidden_dim)
            .init(device);

        Self {
            hidden_dim: config.hidden_dim,
            pred_len: config.pred_len,
            out_channels: config.out_channels,
            num_layers: config.num_layers,
            in_channels: config.in_channels,
            input_proj,
            gs3b_blocks,
            output_proj,
            output_norm,
            temporal_embedding,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, num_nodes, in_channels]
    /// * `adj` - Optional adjacency matrix [num_nodes, num_nodes]
    ///
    /// # Returns
    /// Predictions [batch, pred_len, num_nodes, out_channels]
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        adj: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 4> {
        let [batch_size, seq_len, num_nodes, in_channels] = x.dims();

        // Input projection
        // Reshape: [batch, seq_len, num_nodes, in_channels] -> [batch * seq_len * num_nodes, in_channels]
        let x_flat = x.reshape([batch_size * seq_len * num_nodes, in_channels]);
        let h = self.input_proj.forward(x_flat);

        // Reshape back: [batch * seq_len * num_nodes, hidden_dim] -> [batch, seq_len, num_nodes, hidden_dim]
        let h = h.reshape([batch_size, seq_len, num_nodes, self.hidden_dim]);

        // Add temporal positional embeddings
        let h = self.add_temporal_embeddings(h);

        // Pass through stacked GS3B blocks
        let mut h = h;
        for gs3b in &self.gs3b_blocks {
            h = gs3b.forward(h, adj.clone());
        }

        // Apply final normalization
        let h = self.apply_final_norm(h);

        // Output projection
        // Take the last time step or use all time steps for prediction
        // For now, use last time step: [batch, num_nodes, hidden_dim]
        let h_last: Tensor<B, 3> = h.narrow(1, seq_len - 1, 1).squeeze(1);

        // Reshape: [batch, num_nodes, hidden_dim] -> [batch * num_nodes, hidden_dim]
        let h_last_flat = h_last.reshape([batch_size * num_nodes, self.hidden_dim]);

        // Project to output: [batch * num_nodes, pred_len * out_channels]
        let output = self.output_proj.forward(h_last_flat);

        // Reshape to: [batch, num_nodes, pred_len, out_channels]
        let output = output.reshape([
            batch_size,
            num_nodes,
            self.pred_len,
            self.out_channels,
        ]);

        // Rearrange to: [batch, pred_len, num_nodes, out_channels]
        output.swap_dims(1, 2)
    }

    /// Add temporal positional embeddings
    fn add_temporal_embeddings(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, seq_len, num_nodes, hidden_dim] = x.dims();
        let device = x.device();

        // Create position indices: [0, 1, 2, ..., seq_len-1]
        // Repeat for batch: [batch_size, seq_len]
        let positions = Tensor::<B, 1, burn::tensor::Int>::arange(0..seq_len as i64, &device);
        let positions = positions.unsqueeze().repeat_dim(0, batch_size);

        // Get embeddings: [batch_size, seq_len, hidden_dim]
        let pos_emb = self.temporal_embedding.forward(positions);

        // Expand for nodes: [batch_size, seq_len, 1, hidden_dim]
        let pos_emb = pos_emb.unsqueeze_dim(2);

        // Add to input
        x + pos_emb
    }

    /// Apply final layer normalization
    fn apply_final_norm(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, seq_len, num_nodes, hidden_dim] = x.dims();

        // Reshape: [batch * seq_len * num_nodes, hidden_dim]
        let x_flat = x.reshape([batch_size * seq_len * num_nodes, hidden_dim]);

        // Apply norm
        let normed = self.output_norm.forward(x_flat);

        // Reshape back
        normed.reshape([batch_size, seq_len, num_nodes, hidden_dim])
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        // This would require implementing parameter counting
        // For now, return an estimate
        let params_per_gs3b = self.hidden_dim * self.hidden_dim * 10;
        let total_gs3b_params = params_per_gs3b * self.num_layers;
        let proj_params = self.in_channels * self.hidden_dim
            + self.hidden_dim * self.pred_len * self.out_channels;

        total_gs3b_params + proj_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_stg_mamba_forward() {
        let device = Default::default();
        let config = STGMambaConfig {
            num_nodes: 10,
            in_channels: 3,
            out_channels: 1,
            hidden_dim: 32,
            num_layers: 2,
            state_dim: 16,
            num_granularities: 2,
            expand_factor: 2,
            d_conv: 4,
            dropout: 0.1,
            ffn_expand: 4,
            seq_len: 12,
            pred_len: 6,
        };

        let model = STGMamba::<TestBackend>::new(&config, &device);

        // Input: [batch=2, seq_len=12, num_nodes=10, in_channels=3]
        let x = Tensor::<TestBackend, 4>::random(
            [2, 12, 10, 3],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = model.forward(x, None);

        // Output: [batch=2, pred_len=6, num_nodes=10, out_channels=1]
        assert_eq!(output.dims(), [2, 6, 10, 1]);
    }

    #[test]
    fn test_stg_mamba_financial_config() {
        let config = STGMambaConfig::financial(50);
        assert_eq!(config.num_nodes, 50);
        assert_eq!(config.in_channels, 5);
        assert_eq!(config.out_channels, 1);
    }
}
