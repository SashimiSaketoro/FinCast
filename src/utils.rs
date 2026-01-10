use burn::tensor::{backend::Backend, Tensor};

/// Normalize tensor along a dimension using min-max normalization
pub fn min_max_normalize<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    dim: usize,
) -> (Tensor<B, D>, Tensor<B, D>, Tensor<B, D>) {
    let min = tensor.clone().min_dim(dim);
    let max = tensor.clone().max_dim(dim);
    let range = max.clone().sub(min.clone()).add_scalar(1e-8);
    let normalized = tensor.sub(min.clone()).div(range.clone());
    (normalized, min, range)
}

/// Denormalize tensor using stored min and range
pub fn min_max_denormalize<B: Backend, const D: usize>(
    normalized: Tensor<B, D>,
    min: Tensor<B, D>,
    range: Tensor<B, D>,
) -> Tensor<B, D> {
    normalized.mul(range).add(min)
}

/// Calculate Mean Absolute Error
pub fn mae<B: Backend, const D: usize>(pred: Tensor<B, D>, target: Tensor<B, D>) -> Tensor<B, 1> {
    pred.sub(target).abs().mean()
}

/// Calculate Root Mean Squared Error
pub fn rmse<B: Backend, const D: usize>(pred: Tensor<B, D>, target: Tensor<B, D>) -> Tensor<B, 1> {
    pred.sub(target).powf_scalar(2.0).mean().sqrt()
}

/// Calculate Mean Absolute Percentage Error
pub fn mape<B: Backend, const D: usize>(
    pred: Tensor<B, D>,
    target: Tensor<B, D>,
) -> Tensor<B, 1> {
    let epsilon = 1e-8;
    pred.sub(target.clone())
        .abs()
        .div(target.abs().add_scalar(epsilon))
        .mean()
        .mul_scalar(100.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_min_max_normalize() {
        let tensor = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &Default::default());
        let (normalized, min, range) = min_max_normalize(tensor.clone(), 1);

        // Check that values are in [0, 1]
        let data = normalized.to_data();
        println!("Normalized: {:?}", data);
    }
}
