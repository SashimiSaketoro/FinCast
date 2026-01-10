// Kalman Filtering Graph Neural Network (KFGN)
//
// Part of STG-Mamba architecture
//
// KFGN dynamically integrates and upgrades the STG embeddings from different
// temporal granularities through a learnable Kalman Filtering approach.
//
// Components:
// 1. DynamicFilter-GNN: Generates input-specific dynamic graph structures
// 2. KF-Upgrading: Integrates embeddings through Kalman Filtering
//
// The KF-Upgrading mechanism models STG inputs from different temporal
// granularities as parallel streams, and outputs are integrated through
// statistical theory learning of Kalman Filtering for optimization.

use burn::{
    config::Config,
    module::Module,
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

use super::graph_conv::{DynamicFilterGNN, DynamicFilterGNNConfig};

/// Configuration for KFGN module
#[derive(Config, Debug)]
pub struct KFGNConfig {
    /// Number of nodes in the graph
    pub num_nodes: usize,

    /// Hidden dimension
    pub hidden_dim: usize,

    /// Number of temporal granularities
    #[config(default = 3)]
    pub num_granularities: usize,

    /// Number of GCN layers in DynamicFilter-GNN
    #[config(default = 2)]
    pub num_gcn_layers: usize,

    /// Dropout rate
    #[config(default = 0.1)]
    pub dropout: f64,
}

/// Kalman Filtering Graph Neural Network
///
/// Integrates spatial-temporal graph embeddings using Kalman filtering principles
#[derive(Module, Debug)]
pub struct KFGN<B: Backend> {
    num_nodes: usize,
    hidden_dim: usize,
    num_granularities: usize,

    /// DynamicFilter-GNN for each temporal granularity
    dynamic_gnns: Vec<DynamicFilterGNN<B>>,

    /// Kalman Filter parameters
    /// Process noise covariance (Q) - learned
    q_proj: Linear<B>,

    /// Measurement noise covariance (R) - learned
    r_proj: Linear<B>,

    /// State transition projections for each granularity
    state_transitions: Vec<Linear<B>>,

    /// Integration layer
    integration: Linear<B>,

    /// Layer normalization
    norm: LayerNorm<B>,
}

impl<B: Backend> KFGN<B> {
    /// Create a new KFGN module
    pub fn new(config: &KFGNConfig, device: &B::Device) -> Self {
        // Create DynamicFilter-GNN for each temporal granularity
        let mut dynamic_gnns = Vec::new();
        for _ in 0..config.num_granularities {
            let gnn_config = DynamicFilterGNNConfig {
                num_nodes: config.num_nodes,
                hidden_dim: config.hidden_dim,
                num_layers: config.num_gcn_layers,
                learnable_adj: true,
            };
            dynamic_gnns.push(DynamicFilterGNN::new(&gnn_config, device));
        }

        // Kalman filter parameter projections
        let q_proj = LinearConfig::new(config.hidden_dim, config.hidden_dim)
            .with_bias(true)
            .init(device);

        let r_proj = LinearConfig::new(config.hidden_dim, config.hidden_dim)
            .with_bias(true)
            .init(device);

        // State transition matrices for each granularity
        let mut state_transitions = Vec::new();
        for _ in 0..config.num_granularities {
            state_transitions.push(
                LinearConfig::new(config.hidden_dim, config.hidden_dim)
                    .with_bias(false)
                    .init(device),
            );
        }

        // Integration layer to combine multi-granularity features
        let integration = LinearConfig::new(
            config.hidden_dim * config.num_granularities,
            config.hidden_dim,
        )
        .with_bias(true)
        .init(device);

        // Layer normalization
        let norm = LayerNormConfig::new(config.hidden_dim).init(device);

        Self {
            num_nodes: config.num_nodes,
            hidden_dim: config.hidden_dim,
            num_granularities: config.num_granularities,
            dynamic_gnns,
            q_proj,
            r_proj,
            state_transitions,
            integration,
            norm,
        }
    }

    /// Forward pass through KFGN
    ///
    /// # Arguments
    /// * `x` - Input features [batch, seq_len, num_nodes, hidden_dim]
    /// * `adj` - Optional static adjacency matrix [num_nodes, num_nodes]
    ///
    /// # Returns
    /// Filtered and integrated features [batch, seq_len, num_nodes, hidden_dim]
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        adj: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 4> {
        let [batch_size, seq_len, num_nodes, hidden_dim] = x.dims();

        // Process each temporal granularity
        let mut granularity_outputs = Vec::new();

        for (i, gnn) in self.dynamic_gnns.iter().enumerate() {
            // Apply temporal downsampling/pooling for different granularities
            let x_granular = if i == 0 {
                // Fine-grained: use original sequence
                x.clone()
            } else {
                // Coarser granularities: pool over time
                self.temporal_pool(x.clone(), i + 1)
            };

            let [_, gran_seq_len, _, _] = x_granular.dims();

            // Apply DynamicFilter-GNN to each time step
            let mut time_outputs = Vec::new();
            for t in 0..gran_seq_len {
                let x_t = x_granular.clone().narrow(1, t, 1).squeeze(1); // [batch, num_nodes, hidden_dim]
                let h_t = gnn.forward(x_t, adj.clone()); // [batch, num_nodes, hidden_dim]
                time_outputs.push(h_t.unsqueeze_dim(1)); // [batch, 1, num_nodes, hidden_dim]
            }

            let h_granular = Tensor::cat(time_outputs, 1); // [batch, gran_seq_len, num_nodes, hidden_dim]

            // Upsample back to original sequence length if needed
            let h_granular = if gran_seq_len != seq_len {
                self.temporal_upsample(h_granular, seq_len)
            } else {
                h_granular
            };

            granularity_outputs.push(h_granular);
        }

        // Apply Kalman filtering to integrate multi-granularity features
        let integrated = self.kalman_filter(granularity_outputs);

        // Layer normalization
        self.apply_layer_norm(integrated)
    }

    /// Temporal pooling for coarser granularities
    ///
    /// Applies average pooling over time with stride = granularity
    fn temporal_pool(&self, x: Tensor<B, 4>, granularity: usize) -> Tensor<B, 4> {
        let [batch_size, seq_len, num_nodes, hidden_dim] = x.dims();

        // Calculate new sequence length
        let new_seq_len = (seq_len + granularity - 1) / granularity;

        let mut pooled = Vec::new();
        for i in 0..new_seq_len {
            let start = i * granularity;
            let end = (start + granularity).min(seq_len);
            let window_size = end - start;

            // Extract window and average
            let window = x.clone().narrow(1, start, window_size);
            let averaged = window.mean_dim(1); // [batch, num_nodes, hidden_dim]
            pooled.push(averaged.unsqueeze_dim(1)); // [batch, 1, num_nodes, hidden_dim]
        }

        Tensor::cat(pooled, 1) // [batch, new_seq_len, num_nodes, hidden_dim]
    }

    /// Temporal upsampling using linear interpolation
    fn temporal_upsample(&self, x: Tensor<B, 4>, target_len: usize) -> Tensor<B, 4> {
        let [batch_size, seq_len, num_nodes, hidden_dim] = x.dims();

        if seq_len == target_len {
            return x;
        }

        // Simple repeat-based upsampling
        let repeat_factor = (target_len + seq_len - 1) / seq_len;
        let mut upsampled = Vec::new();

        for t in 0..seq_len {
            let x_t = x.clone().narrow(1, t, 1); // [batch, 1, num_nodes, hidden_dim]
            for _ in 0..repeat_factor {
                if upsampled.len() < target_len {
                    upsampled.push(x_t.clone());
                }
            }
        }

        Tensor::cat(upsampled, 1).narrow(1, 0, target_len) // [batch, target_len, num_nodes, hidden_dim]
    }

    /// Kalman filter integration
    ///
    /// Integrates features from multiple temporal granularities using
    /// learned Kalman filtering parameters
    fn kalman_filter(&self, granularity_features: Vec<Tensor<B, 4>>) -> Tensor<B, 4> {
        let [batch_size, seq_len, num_nodes, hidden_dim] = granularity_features[0].dims();
        let device = granularity_features[0].device();

        // Initialize state estimate
        let mut state = Tensor::<B, 4>::zeros([batch_size, seq_len, num_nodes, hidden_dim], &device);

        // Initialize covariance (as identity initially)
        let mut p = Tensor::<B, 4>::ones([batch_size, seq_len, num_nodes, hidden_dim], &device);

        // Process each granularity as a measurement
        for (i, measurement) in granularity_features.iter().enumerate() {
            // State transition: x_pred = F * x
            let state_pred = self.state_transitions[i].forward(
                state.clone().reshape([batch_size * seq_len * num_nodes, hidden_dim]),
            ).reshape([batch_size, seq_len, num_nodes, hidden_dim]);

            // Process noise: Q
            let q = self.q_proj.forward(
                state_pred.clone().reshape([batch_size * seq_len * num_nodes, hidden_dim]),
            ).reshape([batch_size, seq_len, num_nodes, hidden_dim]);
            let q = q.abs() + 1e-6; // Ensure positive

            // Predicted covariance: P_pred = F P F^T + Q
            let p_pred = p.clone() + q;

            // Measurement noise: R
            let r = self.r_proj.forward(
                measurement.clone().reshape([batch_size * seq_len * num_nodes, hidden_dim]),
            ).reshape([batch_size, seq_len, num_nodes, hidden_dim]);
            let r = r.abs() + 1e-6; // Ensure positive

            // Kalman gain: K = P_pred / (P_pred + R)
            let k = p_pred.clone() / (p_pred.clone() + r);

            // Update state: x = x_pred + K * (z - x_pred)
            let innovation = measurement.clone() - state_pred.clone();
            state = state_pred + k.clone() * innovation;

            // Update covariance: P = (I - K) * P_pred
            let ones = Tensor::<B, 4>::ones([batch_size, seq_len, num_nodes, hidden_dim], &device);
            p = (ones - k) * p_pred;
        }

        state
    }

    /// Apply layer normalization
    fn apply_layer_norm(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, seq_len, num_nodes, hidden_dim] = x.dims();

        // Reshape for layer norm
        let x_reshaped = x.reshape([batch_size * seq_len * num_nodes, hidden_dim]);

        // Apply norm
        let normed = self.norm.forward(x_reshaped);

        // Reshape back
        normed.reshape([batch_size, seq_len, num_nodes, hidden_dim])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_kfgn_forward() {
        let device = Default::default();
        let config = KFGNConfig {
            num_nodes: 10,
            hidden_dim: 32,
            num_granularities: 3,
            num_gcn_layers: 2,
            dropout: 0.1,
        };

        let kfgn = KFGN::<TestBackend>::new(&config, &device);

        // Input: [batch=2, seq_len=12, num_nodes=10, hidden_dim=32]
        let x = Tensor::<TestBackend, 4>::random(
            [2, 12, 10, 32],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = kfgn.forward(x, None);

        assert_eq!(output.dims(), [2, 12, 10, 32]);
    }
}
