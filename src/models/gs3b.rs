// Graph Selective State Space Block (GS3B)
//
// The core building block of STG-Mamba
//
// Components:
// 1. KFGN: Kalman Filtering Graph Neural Network for spatial processing
// 2. ST-S3M: Spatial-Temporal Selective State Space Module (Mamba) for temporal processing
// 3. Feed-forward connection for residual integration
//
// Architecture:
// Input -> KFGN (spatial) -> ST-S3M (temporal) -> FFN -> Output
//                |_________________________________^
//                         Residual Connection

use burn::{
    config::Config,
    module::Module,
    nn::{
        Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
    },
    tensor::{activation, backend::Backend, Tensor},
};

use crate::layers::{
    kfgn::{KFGN, KFGNConfig},
    mamba::{Mamba, MambaConfig},
};

/// Configuration for Graph Selective State Space Block
#[derive(Config, Debug)]
pub struct GS3BConfig {
    /// Number of nodes in the graph
    pub num_nodes: usize,

    /// Hidden dimension
    pub hidden_dim: usize,

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
}

/// Graph Selective State Space Block
///
/// Combines KFGN for spatial processing and Mamba (ST-S3M) for temporal processing
#[derive(Module, Debug)]
pub struct GS3B<B: Backend> {
    num_nodes: usize,
    hidden_dim: usize,

    /// Kalman Filtering Graph Neural Network (spatial processing)
    kfgn: KFGN<B>,

    /// Spatial-Temporal Selective State Space Module (Mamba for temporal processing)
    st_s3m: Mamba<B>,

    /// Feed-forward network
    ffn: FeedForward<B>,

    /// Layer normalizations
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    norm3: LayerNorm<B>,

    /// Dropout
    dropout: Dropout,
}

impl<B: Backend> GS3B<B> {
    /// Create a new GS3B block
    pub fn new(config: &GS3BConfig, device: &B::Device) -> Self {
        // KFGN for spatial processing
        let kfgn_config = KFGNConfig {
            num_nodes: config.num_nodes,
            hidden_dim: config.hidden_dim,
            num_granularities: config.num_granularities,
            num_gcn_layers: 2,
            dropout: config.dropout,
        };
        let kfgn = KFGN::new(&kfgn_config, device);

        // Mamba for temporal processing (ST-S3M)
        let mamba_config = MambaConfig {
            d_model: config.hidden_dim * config.num_nodes,
            d_state: config.state_dim,
            expand: config.expand_factor,
            d_conv: config.d_conv,
            dt_min: 0.001,
            dt_max: 0.1,
            bias: false,
        };
        let st_s3m = Mamba::new(&mamba_config, device);

        // Feed-forward network
        let ffn_config = FeedForwardConfig {
            hidden_dim: config.hidden_dim,
            ffn_dim: config.hidden_dim * config.ffn_expand,
            dropout: config.dropout,
        };
        let ffn = FeedForward::new(&ffn_config, device);

        // Layer normalizations
        let norm1 = LayerNormConfig::new(config.hidden_dim).init(device);
        let norm2 = LayerNormConfig::new(config.hidden_dim * config.num_nodes).init(device);
        let norm3 = LayerNormConfig::new(config.hidden_dim).init(device);

        // Dropout
        let dropout = DropoutConfig::new(config.dropout).init();

        Self {
            num_nodes: config.num_nodes,
            hidden_dim: config.hidden_dim,
            kfgn,
            st_s3m,
            ffn,
            norm1,
            norm2,
            norm3,
            dropout,
        }
    }

    /// Forward pass through GS3B block
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, num_nodes, hidden_dim]
    /// * `adj` - Optional adjacency matrix [num_nodes, num_nodes]
    ///
    /// # Returns
    /// Output tensor [batch, seq_len, num_nodes, hidden_dim]
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        adj: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 4> {
        let [batch_size, seq_len, num_nodes, hidden_dim] = x.dims();

        // Pre-norm + KFGN (spatial processing) + residual
        let normed_x = self.apply_node_norm(x.clone(), &self.norm1);
        let spatial_out = self.kfgn.forward(normed_x, adj);
        let x = x + self.dropout.forward(spatial_out);

        // Pre-norm + ST-S3M (temporal processing) + residual
        let x_flat = x.clone().reshape([batch_size, seq_len, num_nodes * hidden_dim]);
        let normed_x_flat = self.norm2.forward(x_flat.clone());
        let temporal_out = self.st_s3m.forward(normed_x_flat);
        let temporal_out = temporal_out.reshape([batch_size, seq_len, num_nodes, hidden_dim]);
        let x = x + self.dropout.forward(temporal_out);

        // Pre-norm + FFN + residual
        let normed_x = self.apply_node_norm(x.clone(), &self.norm3);
        let ffn_out = self.apply_ffn_to_nodes(normed_x);
        let x = x + self.dropout.forward(ffn_out);

        x
    }

    /// Apply layer norm to each node independently
    fn apply_node_norm(&self, x: Tensor<B, 4>, norm: &LayerNorm<B>) -> Tensor<B, 4> {
        let [batch_size, seq_len, num_nodes, hidden_dim] = x.dims();

        // Reshape to [batch * seq_len * num_nodes, hidden_dim]
        let x_reshaped = x.reshape([batch_size * seq_len * num_nodes, hidden_dim]);

        // Apply norm
        let normed = norm.forward(x_reshaped);

        // Reshape back
        normed.reshape([batch_size, seq_len, num_nodes, hidden_dim])
    }

    /// Apply FFN to each node independently
    fn apply_ffn_to_nodes(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, seq_len, num_nodes, hidden_dim] = x.dims();

        // Reshape to [batch * seq_len * num_nodes, hidden_dim]
        let x_reshaped = x.reshape([batch_size * seq_len * num_nodes, hidden_dim]);

        // Apply FFN
        let out = self.ffn.forward(x_reshaped);

        // Reshape back
        out.reshape([batch_size, seq_len, num_nodes, hidden_dim])
    }
}

/// Feed-Forward Network
#[derive(Config, Debug)]
pub struct FeedForwardConfig {
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub dropout: f64,
}

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(config: &FeedForwardConfig, device: &B::Device) -> Self {
        let fc1 = LinearConfig::new(config.hidden_dim, config.ffn_dim)
            .with_bias(true)
            .init(device);

        let fc2 = LinearConfig::new(config.ffn_dim, config.hidden_dim)
            .with_bias(true)
            .init(device);

        let dropout = DropoutConfig::new(config.dropout).init();

        Self { fc1, fc2, dropout }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = activation::gelu(x);
        let x = self.dropout.forward(x);
        self.fc2.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_gs3b_forward() {
        let device = Default::default();
        let config = GS3BConfig {
            num_nodes: 10,
            hidden_dim: 32,
            state_dim: 16,
            num_granularities: 3,
            expand_factor: 2,
            d_conv: 4,
            dropout: 0.1,
            ffn_expand: 4,
        };

        let gs3b = GS3B::<TestBackend>::new(&config, &device);

        // Input: [batch=2, seq_len=12, num_nodes=10, hidden_dim=32]
        let x = Tensor::<TestBackend, 4>::random(
            [2, 12, 10, 32],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = gs3b.forward(x, None);

        assert_eq!(output.dims(), [2, 12, 10, 32]);
    }
}
