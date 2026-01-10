// Temporal Attention and Processing Modules
//
// Provides temporal modeling capabilities complementary to Mamba

use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Dropout, DropoutConfig},
    tensor::{activation, backend::Backend, Tensor},
};

/// Configuration for Temporal Attention
#[derive(Config, Debug)]
pub struct TemporalAttentionConfig {
    /// Hidden dimension
    pub hidden_dim: usize,

    /// Number of attention heads
    #[config(default = 4)]
    pub num_heads: usize,

    /// Dropout rate
    #[config(default = 0.1)]
    pub dropout: f64,
}

/// Temporal Attention Layer
///
/// Applies multi-head attention along the temporal dimension
#[derive(Module, Debug)]
pub struct TemporalAttention<B: Backend> {
    hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,

    /// Query, Key, Value projections
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,

    /// Output projection
    out_proj: Linear<B>,

    /// Dropout
    dropout: Dropout,
}

impl<B: Backend> TemporalAttention<B> {
    /// Create a new temporal attention layer
    pub fn new(config: &TemporalAttentionConfig, device: &B::Device) -> Self {
        assert!(
            config.hidden_dim % config.num_heads == 0,
            "hidden_dim must be divisible by num_heads"
        );

        let head_dim = config.hidden_dim / config.num_heads;

        let q_proj = LinearConfig::new(config.hidden_dim, config.hidden_dim)
            .with_bias(false)
            .init(device);

        let k_proj = LinearConfig::new(config.hidden_dim, config.hidden_dim)
            .with_bias(false)
            .init(device);

        let v_proj = LinearConfig::new(config.hidden_dim, config.hidden_dim)
            .with_bias(false)
            .init(device);

        let out_proj = LinearConfig::new(config.hidden_dim, config.hidden_dim)
            .with_bias(false)
            .init(device);

        let dropout = DropoutConfig::new(config.dropout).init();

        Self {
            hidden_dim: config.hidden_dim,
            num_heads: config.num_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            dropout,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, hidden_dim]
    ///
    /// # Returns
    /// Attention output [batch, seq_len, hidden_dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = x.dims();

        // Project to Q, K, V
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // Reshape for multi-head attention
        // [batch, seq_len, hidden_dim] -> [batch, num_heads, seq_len, head_dim]
        let q = self.reshape_for_attention(q, batch_size, seq_len);
        let k = self.reshape_for_attention(k, batch_size, seq_len);
        let v = self.reshape_for_attention(v, batch_size, seq_len);

        // Compute attention scores
        // Q @ K^T / sqrt(head_dim)
        let k_t = k.swap_dims(2, 3); // [batch, num_heads, head_dim, seq_len]
        let scores = q.matmul(k_t) / (self.head_dim as f64).sqrt(); // [batch, num_heads, seq_len, seq_len]

        // Apply softmax
        let attn_weights = activation::softmax(scores, 3);

        // Apply dropout
        let attn_weights = self.dropout.forward(attn_weights);

        // Apply attention to values
        let output = attn_weights.matmul(v); // [batch, num_heads, seq_len, head_dim]

        // Reshape back
        // [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_dim]
        let output = self.reshape_from_attention(output, batch_size, seq_len);

        // Output projection
        self.out_proj.forward(output)
    }

    /// Reshape tensor for multi-head attention
    fn reshape_for_attention(
        &self,
        x: Tensor<B, 3>,
        batch_size: usize,
        seq_len: usize,
    ) -> Tensor<B, 4> {
        // [batch, seq_len, hidden_dim] -> [batch, seq_len, num_heads, head_dim]
        let x = x.reshape([batch_size, seq_len, self.num_heads, self.head_dim]);

        // [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        x.swap_dims(1, 2)
    }

    /// Reshape tensor back from multi-head attention
    fn reshape_from_attention(
        &self,
        x: Tensor<B, 4>,
        batch_size: usize,
        seq_len: usize,
    ) -> Tensor<B, 3> {
        // [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
        let x = x.swap_dims(1, 2);

        // [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, hidden_dim]
        x.reshape([batch_size, seq_len, self.hidden_dim])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_temporal_attention() {
        let device = Default::default();
        let config = TemporalAttentionConfig {
            hidden_dim: 64,
            num_heads: 4,
            dropout: 0.1,
        };

        let attention = TemporalAttention::<TestBackend>::new(&config, &device);

        let x = Tensor::<TestBackend, 3>::random(
            [2, 10, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = attention.forward(x);

        assert_eq!(output.dims(), [2, 10, 64]);
    }
}
