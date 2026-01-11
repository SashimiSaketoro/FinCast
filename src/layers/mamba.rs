// Mamba: Selective State Space Model
//
// Based on "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
// Paper: https://arxiv.org/abs/2312.00752
//
// Key equations:
// 1. Discretization (Zero-Order Hold):
//    A̅ = exp(∆A)
//    B̅ = (∆A)^{-1}(exp(∆A) - I)∆B ≈ ∆B (simplified)
//
// 2. State Space Model:
//    h_t = A̅ h_{t-1} + B̅ x_t
//    y_t = C h_t
//
// 3. Selective mechanism:
//    B, C, ∆ = functions of input x

use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        Linear, LinearConfig, PaddingConfig1d,
    },
    tensor::{activation, backend::Backend, Tensor},
};

/// Configuration for Mamba block
#[derive(Config, Debug)]
pub struct MambaConfig {
    /// Input dimension (d_model)
    pub d_model: usize,

    /// State space dimension (d_state or N)
    pub d_state: usize,

    /// Expansion factor for hidden dimension
    pub expand: usize,

    /// Convolution kernel size (d_conv)
    pub d_conv: usize,

    /// Time step initialization range
    #[config(default = 0.001)]
    pub dt_min: f64,

    #[config(default = 0.1)]
    pub dt_max: f64,

    /// Whether to use bias in projections
    #[config(default = false)]
    pub bias: bool,
}

impl MambaConfig {
    /// Initialize a new Mamba configuration with defaults
    pub fn with_defaults(d_model: usize, d_state: usize) -> Self {
        Self {
            d_model,
            d_state,
            expand: 2,
            d_conv: 4,
            dt_min: 0.001,
            dt_max: 0.1,
            bias: false,
        }
    }
}

/// Mamba selective state space model block
#[derive(Module, Debug)]
pub struct Mamba<B: Backend> {
    /// Input dimension
    d_model: usize,

    /// State dimension
    d_state: usize,

    /// Expanded dimension (d_inner = expand * d_model)
    d_inner: usize,

    /// Convolution kernel size
    d_conv: usize,

    /// Time step range
    dt_min: f64,
    dt_max: f64,

    // Projections
    /// Input projection: [d_model -> 2*d_inner] for x and z (gating)
    in_proj: Linear<B>,

    /// Causal 1D convolution
    conv1d: Conv1d<B>,

    /// Projection for generating time step ∆
    x_proj: Linear<B>,

    /// Projection for generating B parameter
    dt_proj: Linear<B>,

    /// Output projection: [d_inner -> d_model]
    out_proj: Linear<B>,

    /// Fixed A matrix (S4D-Real initialization: A_n = -(n+1))
    /// Shape: [d_inner, d_state]
    a_log: Tensor<B, 2>,

    /// Fixed D skip connection parameter
    /// Shape: [d_inner]
    d: Tensor<B, 1>,
}

impl<B: Backend> Mamba<B> {
    /// Initialize a new Mamba block
    pub fn new(config: &MambaConfig, device: &B::Device) -> Self {
        let d_inner = config.expand * config.d_model;

        // Input projection (to x and z)
        let in_proj = LinearConfig::new(config.d_model, d_inner * 2)
            .with_bias(config.bias)
            .init(device);

        // Causal 1D convolution
        let conv1d = Conv1dConfig::new(d_inner, d_inner, config.d_conv)
            .with_padding(PaddingConfig1d::Explicit(config.d_conv - 1))
            .with_groups(d_inner) // Depthwise convolution
            .with_bias(true)
            .init(device);

        // Projections for selective parameters
        let x_proj = LinearConfig::new(d_inner, config.d_state + config.d_state + 1)
            .with_bias(false)
            .init(device);

        // Time step projection with special initialization
        let dt_proj = LinearConfig::new(config.d_state, d_inner)
            .with_bias(true)
            .init(device);

        // Output projection
        let out_proj = LinearConfig::new(d_inner, config.d_model)
            .with_bias(config.bias)
            .init(device);

        // Initialize A matrix with S4D-Real: A_n = -(n+1)
        // Shape: [d_inner, d_state]
        let a_init: Vec<f32> = (0..d_inner)
            .flat_map(|_| {
                (0..config.d_state)
                    .map(|n| -((n + 1) as f32))
            })
            .collect();

        let a_log = Tensor::<B, 2>::from_floats(a_init.as_slice(), device)
            .reshape([d_inner, config.d_state])
            .require_grad();

        // Initialize D (skip connection)
        let d = Tensor::<B, 1>::ones([d_inner], device).require_grad();

        Self {
            d_model: config.d_model,
            d_state: config.d_state,
            d_inner,
            d_conv: config.d_conv,
            dt_min: config.dt_min,
            dt_max: config.dt_max,
            in_proj,
            conv1d,
            x_proj,
            dt_proj,
            out_proj,
            a_log,
            d,
        }
    }

    /// Forward pass through Mamba block
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, seq_len, d_model]
    ///
    /// # Returns
    /// Output tensor of shape [batch, seq_len, d_model]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _d_model] = x.dims();

        // Input projection
        // x_and_res: [batch, seq_len, 2*d_inner]
        let x_and_res = self.in_proj.forward(x);

        // Split into x and residual (for gating)
        // x: [batch, seq_len, d_inner]
        // res: [batch, seq_len, d_inner]
        let chunks = x_and_res.chunk(2, 2);
        let x = chunks[0].clone();
        let res = chunks[1].clone();

        // Transpose for conv1d: [batch, d_inner, seq_len]
        let x = x.swap_dims(1, 2);

        // Causal convolution
        let x = self.conv1d.forward(x);

        // Remove padding (we added d_conv-1 padding)
        let x = x.narrow(2, 0, seq_len);

        // Transpose back: [batch, seq_len, d_inner]
        let x = x.swap_dims(1, 2);

        // Activation
        let x = activation::silu(x);

        // Selective SSM
        let y = self.selective_scan(x.clone());

        // Gating mechanism (SiLU gating)
        let y = y * activation::silu(res);

        // Output projection
        self.out_proj.forward(y)
    }

    /// Selective scan using state space model
    ///
    /// Implements the core SSM computation with input-dependent parameters
    fn selective_scan(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, d_inner] = x.dims();

        // Generate input-dependent parameters B, C, ∆
        // x_proj output: [batch, seq_len, d_state + d_state + 1]
        let x_dbl = self.x_proj.forward(x.clone());

        // Split into delta, B, C
        // delta: [batch, seq_len, 1]
        // B: [batch, seq_len, d_state]
        // C: [batch, seq_len, d_state]
        let delta = x_dbl.clone().narrow(2, 0, 1);
        let b = x_dbl.clone().narrow(2, 1, self.d_state);
        let c = x_dbl.narrow(2, 1 + self.d_state, self.d_state);

        // Process time step: ensure it's in [dt_min, dt_max]
        let delta = activation::softplus(delta, 1.0); // Make positive
        let delta = delta.clamp_min(self.dt_min);
        let delta = delta.clamp_max(self.dt_max);

        // Project delta to get per-channel time steps
        // dt_proj: [d_state -> d_inner]
        let delta: Tensor<B, 2> = delta.squeeze(2); // [batch, seq_len]
        let delta = self.dt_proj.forward(delta); // [batch, seq_len, d_inner]

        // Get A matrix (convert from log space)
        // A: [d_inner, d_state]
        let a = self.a_log.clone().exp().neg();

        // Discretize using Zero-Order Hold
        // A̅ = exp(∆A)
        // B̅ = ∆B (simplified version)

        // Expand dimensions for broadcasting
        // delta: [batch, seq_len, d_inner, 1]
        // a: [1, 1, d_inner, d_state]
        let delta_expanded = delta.clone().unsqueeze_dim(3);
        let a_expanded: Tensor<B, 3> = a.clone().unsqueeze_dim(0);
        let a_expanded: Tensor<B, 4> = a_expanded.unsqueeze_dim(0);

        // Discretize A: A̅ = exp(∆A)
        // delta_a: [batch, seq_len, d_inner, d_state]
        let delta_a = delta_expanded.clone() * a_expanded;
        let a_bar = delta_a.clone().exp(); // [batch, seq_len, d_inner, d_state]

        // Discretize B: B̅ = ∆B (simplified)
        // b: [batch, seq_len, d_state] -> [batch, seq_len, 1, d_state]
        // delta: [batch, seq_len, d_inner, 1]
        let b_expanded = b.unsqueeze_dim(2); // [batch, seq_len, 1, d_state]
        let b_bar = delta_expanded * b_expanded; // [batch, seq_len, d_inner, d_state]

        // Run SSM recurrence
        // h_t = A̅ * h_{t-1} + B̅ * x_t
        // y_t = C * h_t + D * x_t

        let y = self.ssm_recurrence(x.clone(), a_bar, b_bar, c);

        // Add skip connection: D * x
        let d_expanded: Tensor<B, 2> = self.d.clone().unsqueeze_dim(0);
        let d_expanded: Tensor<B, 3> = d_expanded.unsqueeze_dim(0);
        let skip = x * d_expanded;

        y + skip
    }

    /// SSM recurrence computation
    ///
    /// Computes: h_t = A̅ h_{t-1} + B̅ x_t, y_t = C h_t
    fn ssm_recurrence(
        &self,
        x: Tensor<B, 3>,
        a_bar: Tensor<B, 4>,
        b_bar: Tensor<B, 4>,
        c: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, d_inner] = x.dims();

        // Initialize hidden state: [batch, d_inner, d_state]
        let mut h = Tensor::<B, 3>::zeros([batch_size, d_inner, self.d_state], &x.device());

        let mut outputs = Vec::new();

        // Sequential processing over time
        for t in 0..seq_len {
            // Get current inputs
            // x_t: [batch, d_inner]
            let x_t: Tensor<B, 2> = x.clone().narrow(1, t, 1).squeeze(1);

            // a_bar_t: [batch, d_inner, d_state]
            let a_bar_t: Tensor<B, 3> = a_bar.clone().narrow(1, t, 1).squeeze(1);

            // b_bar_t: [batch, d_inner, d_state]
            let b_bar_t: Tensor<B, 3> = b_bar.clone().narrow(1, t, 1).squeeze(1);

            // c_t: [batch, d_state]
            let c_t: Tensor<B, 2> = c.clone().narrow(1, t, 1).squeeze(1);

            // State update: h_t = A̅ * h_{t-1} + B̅ * x_t
            // h: [batch, d_inner, d_state]
            // a_bar_t: [batch, d_inner, d_state]
            h = h * a_bar_t + b_bar_t * x_t.clone().unsqueeze_dim(2);

            // Output: y_t = C * h_t
            // h: [batch, d_inner, d_state]
            // c_t: [batch, d_state] -> [batch, 1, d_state]
            let c_t_expanded = c_t.unsqueeze_dim(1);

            // Sum over state dimension
            let y_t = (h.clone() * c_t_expanded).sum_dim(2); // [batch, d_inner]

            outputs.push(y_t.unsqueeze_dim(1)); // [batch, 1, d_inner]
        }

        // Concatenate outputs along time dimension
        Tensor::cat(outputs, 1) // [batch, seq_len, d_inner]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_mamba_forward() {
        let device = Default::default();
        let config = MambaConfig::with_defaults(64, 16);

        let mamba = Mamba::<TestBackend>::new(&config, &device);

        // Input: [batch=2, seq_len=10, d_model=64]
        let x = Tensor::<TestBackend, 3>::random(
            [2, 10, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = mamba.forward(x);

        // Output should have same shape as input
        assert_eq!(output.dims(), [2, 10, 64]);
    }

    #[test]
    fn test_mamba_dimensions() {
        let device = Default::default();
        let config = MambaConfig {
            d_model: 32,
            d_state: 8,
            expand: 2,
            d_conv: 4,
            dt_min: 0.001,
            dt_max: 0.1,
            bias: false,
        };

        let mamba = Mamba::<TestBackend>::new(&config, &device);

        assert_eq!(mamba.d_model, 32);
        assert_eq!(mamba.d_inner, 64); // expand * d_model
        assert_eq!(mamba.d_state, 8);
    }
}
