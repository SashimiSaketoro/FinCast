// Graph Convolution Network (GCN) layers
//
// Implements spatial processing for graph-structured data
// Used in the DynamicFilter-GNN component of STG-Mamba
//
// Standard GCN equation:
// H^{(l+1)} = σ(D^{-1/2} A D^{-1/2} H^{(l)} W^{(l)})
//
// Where:
// - A: Adjacency matrix (with self-loops)
// - D: Degree matrix
// - H: Node features
// - W: Learnable weight matrix
// - σ: Activation function

use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{activation, backend::Backend, Tensor},
};

/// Configuration for Graph Convolution layer
#[derive(Config, Debug)]
pub struct GraphConvConfig {
    /// Input feature dimension
    pub in_features: usize,

    /// Output feature dimension
    pub out_features: usize,

    /// Whether to use bias
    #[config(default = true)]
    pub bias: bool,

    /// Activation function type
    pub activation: String,
}

/// Graph Convolution Layer
///
/// Applies graph convolution operation:
/// H' = σ(A_norm H W)
#[derive(Module, Debug)]
pub struct GraphConv<B: Backend> {
    /// Linear transformation
    linear: Linear<B>,

    /// Activation function name
    activation: String,
}

impl<B: Backend> GraphConv<B> {
    /// Create a new graph convolution layer
    pub fn new(config: &GraphConvConfig, device: &B::Device) -> Self {
        let linear = LinearConfig::new(config.in_features, config.out_features)
            .with_bias(config.bias)
            .init(device);

        Self {
            linear,
            activation: config.activation.clone(),
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Node features [batch, num_nodes, in_features]
    /// * `adj` - Normalized adjacency matrix [num_nodes, num_nodes]
    ///
    /// # Returns
    /// Updated node features [batch, num_nodes, out_features]
    pub fn forward(&self, x: Tensor<B, 3>, adj: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch_size, num_nodes, in_features] = x.dims();

        // Reshape for matrix multiplication
        // x: [batch * num_nodes, in_features]
        let x_reshaped = x.reshape([batch_size * num_nodes, in_features]);

        // Apply linear transformation: H W
        // out: [batch * num_nodes, out_features]
        let out = self.linear.forward(x_reshaped);

        // Reshape back: [batch, num_nodes, out_features]
        let [_, out_features] = out.dims();
        let out = out.reshape([batch_size, num_nodes, out_features]);

        // Apply graph convolution: A H'
        // For each batch: adj @ out[b]
        let mut batch_outputs = Vec::new();
        for b in 0..batch_size {
            let out_b = out.clone().narrow(0, b, 1).squeeze(0); // [num_nodes, out_features]

            // Graph convolution: adj @ out_b
            let conv_out = adj.clone().matmul(out_b); // [num_nodes, out_features]

            batch_outputs.push(conv_out.unsqueeze_dim(0)); // [1, num_nodes, out_features]
        }

        let result = Tensor::cat(batch_outputs, 0); // [batch, num_nodes, out_features]

        // Apply activation
        self.apply_activation(result)
    }

    /// Apply activation function
    fn apply_activation(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self.activation.as_str() {
            "relu" => activation::relu(x),
            "gelu" => activation::gelu(x),
            "silu" => activation::silu(x),
            "tanh" => activation::tanh(x),
            _ => x, // No activation
        }
    }
}

/// Dynamic Filter GNN
///
/// Generates input-specific dynamic graph structures
/// Part of the KFGN module in STG-Mamba
#[derive(Config, Debug)]
pub struct DynamicFilterGNNConfig {
    /// Number of nodes
    pub num_nodes: usize,

    /// Feature dimension
    pub hidden_dim: usize,

    /// Number of graph convolution layers
    #[config(default = 2)]
    pub num_layers: usize,

    /// Whether to learn adjacency matrix
    #[config(default = true)]
    pub learnable_adj: bool,
}

/// Dynamic Filter GNN module
///
/// Generates dynamic, input-dependent adjacency matrices
/// and applies graph convolutions
#[derive(Module, Debug)]
pub struct DynamicFilterGNN<B: Backend> {
    num_nodes: usize,
    hidden_dim: usize,
    learnable_adj: bool,

    /// Graph convolution layers
    gcn_layers: Vec<GraphConv<B>>,

    /// Learnable base adjacency matrix (if enabled)
    /// Shape: [num_nodes, num_nodes]
    base_adj: Option<Tensor<B, 2>>,

    /// Projections for generating dynamic adjacency
    node_embedding: Linear<B>,
}

impl<B: Backend> DynamicFilterGNN<B> {
    /// Create a new Dynamic Filter GNN
    pub fn new(config: &DynamicFilterGNNConfig, device: &B::Device) -> Self {
        let mut gcn_layers = Vec::new();

        // Create GCN layers
        for i in 0..config.num_layers {
            let in_features = if i == 0 {
                config.hidden_dim
            } else {
                config.hidden_dim
            };

            let gcn_config = GraphConvConfig {
                in_features,
                out_features: config.hidden_dim,
                bias: true,
                activation: if i == config.num_layers - 1 {
                    "none".to_string()
                } else {
                    "relu".to_string()
                },
            };

            gcn_layers.push(GraphConv::new(&gcn_config, device));
        }

        // Learnable base adjacency
        let base_adj = if config.learnable_adj {
            Some(
                Tensor::<B, 2>::random(
                    [config.num_nodes, config.num_nodes],
                    burn::tensor::Distribution::Uniform(0.0, 1.0),
                    device,
                )
                .require_grad(),
            )
        } else {
            None
        };

        // Node embedding for dynamic adjacency generation
        let node_embedding = LinearConfig::new(config.hidden_dim, config.hidden_dim)
            .with_bias(false)
            .init(device);

        Self {
            num_nodes: config.num_nodes,
            hidden_dim: config.hidden_dim,
            learnable_adj: config.learnable_adj,
            gcn_layers,
            base_adj,
            node_embedding,
        }
    }

    /// Forward pass with optional static adjacency matrix
    ///
    /// # Arguments
    /// * `x` - Node features [batch, num_nodes, hidden_dim]
    /// * `static_adj` - Optional static adjacency matrix [num_nodes, num_nodes]
    ///
    /// # Returns
    /// Updated features [batch, num_nodes, hidden_dim]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        static_adj: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        // Generate dynamic adjacency matrix
        let adj = self.generate_adjacency(x.clone(), static_adj);

        // Apply GCN layers
        let mut h = x;
        for gcn in &self.gcn_layers {
            h = gcn.forward(h, adj.clone());
        }

        h
    }

    /// Generate dynamic adjacency matrix
    ///
    /// Combines static and learnable adjacency with input-dependent adjustments
    fn generate_adjacency(
        &self,
        x: Tensor<B, 3>,
        static_adj: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 2> {
        let device = x.device();

        // Start with base adjacency
        let mut adj = if let Some(static_adj_tensor) = static_adj {
            static_adj_tensor
        } else if let Some(ref base) = self.base_adj {
            base.clone()
        } else {
            // Identity matrix as fallback
            Tensor::<B, 2>::eye(self.num_nodes, &device)
        };

        // If learnable, add the learnable component
        if let Some(ref base) = self.base_adj {
            adj = adj + base.clone();
        }

        // Normalize adjacency: D^{-1/2} A D^{-1/2}
        adj = self.normalize_adjacency(adj);

        adj
    }

    /// Normalize adjacency matrix
    ///
    /// Applies symmetric normalization: D^{-1/2} A D^{-1/2}
    fn normalize_adjacency(&self, adj: Tensor<B, 2>) -> Tensor<B, 2> {
        let [num_nodes, _] = adj.dims();
        let device = adj.device();

        // Add self-loops
        let eye = Tensor::<B, 2>::eye(num_nodes, &device);
        let adj = adj + eye;

        // Compute degree matrix D
        let degrees = adj.clone().sum_dim(1); // [num_nodes]

        // D^{-1/2}
        let deg_inv_sqrt = degrees.powf_scalar(-0.5);

        // Expand for broadcasting
        let deg_inv_sqrt_expanded = deg_inv_sqrt.clone().unsqueeze_dim(1); // [num_nodes, 1]

        // D^{-1/2} A D^{-1/2}
        // First: D^{-1/2} A
        let adj = adj * deg_inv_sqrt_expanded.clone();

        // Then: (D^{-1/2} A) D^{-1/2}
        let deg_inv_sqrt_expanded_t = deg_inv_sqrt.unsqueeze_dim(0); // [1, num_nodes]
        adj * deg_inv_sqrt_expanded_t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_graph_conv() {
        let device = Default::default();
        let config = GraphConvConfig {
            in_features: 32,
            out_features: 64,
            bias: true,
            activation: "relu".to_string(),
        };

        let gcn = GraphConv::<TestBackend>::new(&config, &device);

        // Create input
        let x = Tensor::<TestBackend, 3>::random(
            [2, 10, 32],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        // Create adjacency matrix
        let adj = Tensor::<TestBackend, 2>::eye(10, &device);

        let output = gcn.forward(x, adj);

        assert_eq!(output.dims(), [2, 10, 64]);
    }

    #[test]
    fn test_dynamic_filter_gnn() {
        let device = Default::default();
        let config = DynamicFilterGNNConfig {
            num_nodes: 10,
            hidden_dim: 32,
            num_layers: 2,
            learnable_adj: true,
        };

        let gnn = DynamicFilterGNN::<TestBackend>::new(&config, &device);

        let x = Tensor::<TestBackend, 3>::random(
            [2, 10, 32],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = gnn.forward(x, None);

        assert_eq!(output.dims(), [2, 10, 32]);
    }
}
