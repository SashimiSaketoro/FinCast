# STG-Mamba: Spatial-Temporal Graph Learning via Selective State Space Model

A Rust implementation of STG-Mamba using the [Burn](https://burn.dev) deep learning framework for financial time-series prediction and spatial-temporal forecasting.

## Overview

STG-Mamba is a state-of-the-art architecture that combines:
- **Selective State Space Models (Mamba)** for efficient temporal sequence modeling
- **Graph Neural Networks** for capturing spatial relationships
- **Kalman Filtering** for multi-granularity temporal integration

This implementation focuses on architectural accuracy, following the equations and design from the original paper.

## References

- **Paper**: [STG-Mamba: Spatial-Temporal Graph Learning via Selective State Space Model](https://arxiv.org/abs/2403.12418)
- **Original Implementation**: [LincanLi98/STG-Mamba](https://github.com/LincanLi98/STG-Mamba)
- **Mamba Paper**: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)

## Architecture

### Core Components

#### 1. Mamba (Selective State Space Model)

The Mamba layer implements a selective state space model with input-dependent parameters:

**Continuous State Space Model:**
```
h'(t) = A h(t) + B x(t)
y(t) = C h(t) + D x(t)
```

**Discretization (Zero-Order Hold):**
```
A̅ = exp(∆A)
B̅ = ∆B  (simplified approximation)
```

**Selective Mechanism:**
- Parameters B, C, and ∆ (time step) are functions of the input
- Allows the model to selectively propagate or forget information
- **S4D-Real Initialization**: A_n = -(n+1) for stable dynamics

**Key Features:**
- Depthwise causal convolution (kernel size = 4)
- SiLU activation and gating
- Linear time complexity: O(L) where L is sequence length

#### 2. KFGN (Kalman Filtering Graph Neural Network)

Dynamically integrates spatial-temporal graph embeddings from multiple temporal granularities:

**Components:**
- **DynamicFilter-GNN**: Generates input-specific dynamic graph structures
- **KF-Upgrading**: Kalman filtering for multi-granularity integration

**Kalman Filter Equations:**
```
# Prediction
x_pred = F * x
P_pred = F * P * F^T + Q

# Update
K = P_pred / (P_pred + R)
x = x_pred + K * (z - x_pred)
P = (I - K) * P_pred
```

Where:
- F: State transition matrix (learned)
- Q: Process noise covariance (learned)
- R: Measurement noise covariance (learned)
- K: Kalman gain
- z: Measurement (from different temporal granularities)

#### 3. GS3B (Graph Selective State Space Block)

The fundamental building block combining spatial and temporal processing:

```
Input
  ↓
KFGN (Spatial Processing)
  ↓
Mamba/ST-S3M (Temporal Processing)
  ↓
Feed-Forward Network
  ↓
Output
```

Each component includes:
- Pre-layer normalization
- Residual connections
- Dropout for regularization

#### 4. STG-Mamba Model

Stacks multiple GS3B blocks with:
- Input projection from raw features to hidden dimension
- Temporal positional embeddings
- Output projection to prediction targets

## Mathematical Formulation

### Mamba SSM

1. **Input Projection:**
   ```
   x, z = split(W_in * input)
   ```

2. **Causal Convolution:**
   ```
   x = Conv1D_causal(x)
   x = SiLU(x)
   ```

3. **Selective Parameters Generation:**
   ```
   ∆, B, C = split(W_proj * x)
   ∆ = softplus(∆)  # Ensure positive
   ```

4. **Discretization:**
   ```
   A̅ = exp(∆ ⊙ A)
   B̅ = ∆ ⊙ B
   ```

5. **State Space Recurrence:**
   ```
   h_t = A̅_t ⊙ h_{t-1} + B̅_t ⊙ x_t
   y_t = C_t · h_t + D ⊙ x_t
   ```

6. **Gating:**
   ```
   output = y ⊙ SiLU(z)
   ```

### Graph Convolution

**Symmetric Normalization:**
```
H' = σ(D^{-1/2} A D^{-1/2} H W)
```

Where:
- A: Adjacency matrix with self-loops
- D: Degree matrix
- σ: Activation function (ReLU/GELU)

## Implementation Details

### Key Design Choices

1. **Accuracy over Performance**:
   - Sequential SSM recurrence for correctness
   - Can be optimized with parallel scan in future versions

2. **Type Safety**:
   - Extensive use of Rust's type system
   - Compile-time guarantees for tensor dimensions

3. **Modular Architecture**:
   - Each component is independently testable
   - Clean separation of concerns

### Configuration

**For Financial Forecasting (50 assets):**
```rust
let config = STGMambaConfig {
    num_nodes: 50,           // Number of assets
    in_channels: 5,          // OHLCV features
    out_channels: 1,         // Price/return prediction
    hidden_dim: 128,
    num_layers: 4,           // Stacked GS3B blocks
    state_dim: 32,           // SSM state dimension
    num_granularities: 3,    // Temporal scales
    expand_factor: 2,        // Mamba expansion
    d_conv: 4,               // Convolution kernel
    dropout: 0.15,
    ffn_expand: 4,
    seq_len: 60,             // 60 timesteps lookback
    pred_len: 20,            // 20 timesteps ahead
};
```

**Preset Configurations:**
- `STGMambaConfig::pems04()` - PEMS04 traffic dataset (307 nodes)
- `STGMambaConfig::knowair()` - KnowAir pollution dataset (184 nodes)
- `STGMambaConfig::financial(n)` - Financial forecasting (n assets)

## Usage

### Basic Example

```rust
use stg_mamba::{STGMamba, STGMambaConfig, Backend};
use burn::tensor::Tensor;

// Create model
let device = Default::default();
let config = STGMambaConfig::financial(50);
let model = STGMamba::<Backend>::new(&config, &device);

// Input: [batch, seq_len, num_nodes, in_channels]
let x = Tensor::random([32, 60, 50, 5], burn::tensor::Distribution::Normal(0.0, 1.0), &device);

// Optional adjacency matrix: [num_nodes, num_nodes]
let adj = Some(Tensor::eye(50, &device));

// Forward pass
let predictions = model.forward(x, adj);
// Output: [batch, pred_len, num_nodes, out_channels]
// Shape: [32, 20, 50, 1]
```

### Command Line Interface

```bash
# Train model
stg-mamba train --dataset financial_data.csv --epochs 100 --batch-size 32

# Run prediction
stg-mamba predict --model model.pt --input data.csv --output predictions.csv

# Evaluate
stg-mamba evaluate --model model.pt --data test.csv
```

## Architecture Verification

### Key Equations Implemented

✅ **Mamba SSM Discretization**:
- Zero-Order Hold: A̅ = exp(∆A)
- Simplified B̅ = ∆B

✅ **Selective Mechanism**:
- Input-dependent B, C, ∆
- Softplus activation for ∆

✅ **S4D-Real Initialization**:
- A_n = -(n+1) for stability

✅ **Kalman Filtering**:
- State prediction, measurement update
- Learned Q, R covariances

✅ **Graph Convolution**:
- Symmetric normalization
- Dynamic adjacency generation

## Model Complexity

**Time Complexity:**
- Mamba SSM: O(BLD) where B=batch, L=length, D=dimension
- Graph Conv: O(BLN²D) where N=num_nodes
- Overall: O(BLN²D) dominated by graph operations

**Space Complexity:**
- Parameters: ~O(D² * num_layers)
- Activations: O(BLND)

## Building from Source

```bash
# Clone repository
git clone https://github.com/SashimiSaketoro/FinCast
cd FinCast

# Build
cargo build --release

# Run tests
cargo test

# Build documentation
cargo doc --open
```

## Dependencies

- **Burn** v0.16: Deep learning framework
- **ndarray**: Numerical computing
- **serde**: Serialization
- **clap**: CLI parsing

## Future Work

- [ ] Parallel selective scan (hardware-aware)
- [ ] Training pipeline with optimizers
- [ ] Data loaders for financial datasets
- [ ] Model checkpointing and resuming
- [ ] Mixed precision training
- [ ] Distributed training support
- [ ] WebAssembly inference
- [ ] Python bindings

## Verification Against Original

| Component | Implementation Status | Verification |
|-----------|----------------------|--------------|
| Mamba SSM | ✅ Complete | Equations match paper |
| KFGN | ✅ Complete | Multi-granularity KF |
| GS3B | ✅ Complete | Residual architecture |
| Graph Conv | ✅ Complete | Symmetric normalization |
| Full Model | ✅ Complete | Stacked blocks |
| Training | 🚧 Planned | - |
| Data Loading | 🚧 Planned | - |

## License

MIT

## Citation

If you use this implementation, please cite both the original paper and Mamba:

```bibtex
@article{li2024stgmamba,
  title={STG-Mamba: Spatial-Temporal Graph Learning via Selective State Space Model},
  author={Li, Lincan and others},
  journal={arXiv preprint arXiv:2403.12418},
  year={2024}
}

@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```

## Acknowledgments

- Original STG-Mamba authors for the innovative architecture
- Burn team for the excellent deep learning framework
- Mamba authors for the selective state space model

---

**Status**: Core architecture complete ✅ | Training pipeline in progress 🚧

For questions or issues, please open an issue on GitHub.
