// Core indicator calculations

/// Simple Moving Average
pub fn sma(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period {
        return vec![];
    }

    let mut result = Vec::with_capacity(data.len() - period + 1);

    // Calculate first SMA
    let mut sum: f64 = data[..period].iter().sum();
    result.push(sum / period as f64);

    // Rolling calculation
    for i in period..data.len() {
        sum = sum - data[i - period] + data[i];
        result.push(sum / period as f64);
    }

    result
}

/// Exponential Moving Average
pub fn ema(data: &[f64], period: usize) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }

    let mut result = Vec::with_capacity(data.len());
    let multiplier = 2.0 / (period as f64 + 1.0);

    // Start with SMA as first EMA value
    if data.len() >= period {
        let initial_sma: f64 = data[..period].iter().sum::<f64>() / period as f64;

        // Pad with NaN for the first period-1 values
        for _ in 0..period - 1 {
            result.push(f64::NAN);
        }
        result.push(initial_sma);

        // Calculate EMA for remaining values
        for i in period..data.len() {
            let ema_val = (data[i] - result[i - 1]) * multiplier + result[i - 1];
            result.push(ema_val);
        }
    } else {
        // Not enough data, return all NaN
        return vec![f64::NAN; data.len()];
    }

    result
}

/// Standard Deviation
pub fn std_dev(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period {
        return vec![];
    }

    let mut result = Vec::with_capacity(data.len() - period + 1);

    for i in 0..=(data.len() - period) {
        let window = &data[i..i + period];
        let mean = window.iter().sum::<f64>() / period as f64;

        let variance = window
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / period as f64;

        result.push(variance.sqrt());
    }

    result
}

/// Calculate returns (percentage change)
pub fn returns(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 {
        return vec![];
    }

    data.windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect()
}

/// Calculate log returns
pub fn log_returns(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 {
        return vec![];
    }

    data.windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}

/// Calculate typical price (HLC/3)
pub fn typical_price(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    high.iter()
        .zip(low.iter())
        .zip(close.iter())
        .map(|((&h, &l), &c)| (h + l + c) / 3.0)
        .collect()
}

/// Calculate true range
pub fn true_range(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    if high.len() < 2 || low.len() < 2 || close.len() < 2 {
        return vec![];
    }

    let mut result = Vec::with_capacity(high.len() - 1);

    for i in 1..high.len() {
        let tr1 = high[i] - low[i];
        let tr2 = (high[i] - close[i - 1]).abs();
        let tr3 = (low[i] - close[i - 1]).abs();

        result.push(tr1.max(tr2).max(tr3));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&data, 3);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 2.0).abs() < 1e-10);  // (1+2+3)/3
        assert!((result[1] - 3.0).abs() < 1e-10);  // (2+3+4)/3
        assert!((result[2] - 4.0).abs() < 1e-10);  // (3+4+5)/3
    }

    #[test]
    fn test_ema() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ema(&data, 3);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10);  // SMA for first value
    }

    #[test]
    fn test_returns() {
        let data = vec![100.0, 110.0, 105.0];
        let result = returns(&data);

        assert_eq!(result.len(), 2);
        assert!((result[0] - 0.1).abs() < 1e-10);  // 10% gain
        assert!((result[1] - (-0.1 / 1.1)).abs() < 1e-6);  // ~-4.5% loss
    }

    #[test]
    fn test_std_dev() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let result = std_dev(&data, 4);

        assert!(result.len() > 0);
        // First window: [2,4,4,4], mean=3.5, should have some variance
        assert!(result[0] > 0.0);
    }
}
