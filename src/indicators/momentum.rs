// Momentum indicators

use super::core::ema;

/// Relative Strength Index (RSI)
/// Returns values between 0 and 100
pub fn rsi(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period + 1 {
        return vec![];
    }

    let mut gains = Vec::new();
    let mut losses = Vec::new();

    // Calculate price changes
    for i in 1..data.len() {
        let change = data[i] - data[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(change.abs());
        }
    }

    // Calculate average gains and losses
    let mut result = Vec::with_capacity(data.len() - period);

    // First RSI uses SMA
    let mut avg_gain: f64 = gains[..period].iter().sum::<f64>() / period as f64;
    let mut avg_loss: f64 = losses[..period].iter().sum::<f64>() / period as f64;

    let rs = if avg_loss == 0.0 {
        100.0
    } else {
        avg_gain / avg_loss
    };
    result.push(100.0 - (100.0 / (1.0 + rs)));

    // Subsequent RSI uses smoothed averages (Wilder's smoothing)
    for i in period..gains.len() {
        avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
        avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;

        let rs = if avg_loss == 0.0 {
            100.0
        } else {
            avg_gain / avg_loss
        };
        result.push(100.0 - (100.0 / (1.0 + rs)));
    }

    result
}

/// Stochastic Oscillator
/// Returns (K, D) where K is fast stochastic and D is slow stochastic
pub fn stochastic(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    k_period: usize,
    d_period: usize,
) -> Vec<(f64, f64)> {
    if high.len() < k_period || low.len() < k_period || close.len() < k_period || k_period == 0 {
        return vec![];
    }

    let mut k_values = Vec::new();

    // Calculate %K
    for i in k_period - 1..close.len() {
        let start_idx = i.saturating_sub(k_period - 1);
        let window_high = &high[start_idx..=i];
        let window_low = &low[start_idx..=i];

        let highest = window_high.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lowest = window_low.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        let k = if highest == lowest {
            50.0
        } else {
            100.0 * (close[i] - lowest) / (highest - lowest)
        };

        k_values.push(k);
    }

    // Calculate %D (SMA of %K)
    if k_values.len() < d_period || d_period == 0 {
        return vec![];
    }

    let mut result = Vec::new();

    for i in d_period - 1..k_values.len() {
        let start_idx = i.saturating_sub(d_period - 1);
        let window_len = i - start_idx + 1;
        let d = k_values[start_idx..=i].iter().sum::<f64>() / window_len as f64;
        result.push((k_values[i], d));
    }

    result
}

/// Rate of Change (ROC)
pub fn roc(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period + 1 {
        return vec![];
    }

    data.windows(period + 1)
        .map(|w| {
            let old = w[0];
            let new = w[period];
            ((new - old) / old) * 100.0
        })
        .collect()
}

/// Momentum
pub fn momentum(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period + 1 {
        return vec![];
    }

    data.windows(period + 1)
        .map(|w| w[period] - w[0])
        .collect()
}

/// Williams %R
pub fn williams_r(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Vec<f64> {
    if high.len() < period || low.len() < period || close.len() < period {
        return vec![];
    }

    let mut result = Vec::new();

    for i in period - 1..close.len() {
        let start_idx = i.saturating_sub(period - 1);
        let window_high = &high[start_idx..=i];
        let window_low = &low[start_idx..=i];

        let highest = window_high.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lowest = window_low.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        let wr = if highest == lowest {
            -50.0
        } else {
            -100.0 * (highest - close[i]) / (highest - lowest)
        };

        result.push(wr);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsi() {
        let data = vec![
            44.0, 44.25, 44.5, 43.75, 44.0, 44.5, 45.0, 45.25,
            45.5, 45.0, 44.5, 44.0, 43.5, 43.0, 42.5  // Added 15th element
        ];
        let result = rsi(&data, 14);

        assert!(!result.is_empty());
        // RSI should be between 0 and 100
        for &val in &result {
            assert!(val >= 0.0 && val <= 100.0);
        }
    }

    #[test]
    fn test_stochastic() {
        let high = vec![48.0, 48.5, 49.0, 49.5, 50.0];
        let low = vec![46.0, 46.5, 47.0, 47.5, 48.0];
        let close = vec![47.0, 47.5, 48.0, 48.5, 49.0];

        let result = stochastic(&high, &low, &close, 3, 2);

        assert!(!result.is_empty());
        for (k, d) in result {
            assert!(k >= 0.0 && k <= 100.0);
            assert!(d >= 0.0 && d <= 100.0);
        }
    }

    #[test]
    fn test_roc() {
        let data = vec![100.0, 110.0, 105.0, 115.0];
        let result = roc(&data, 2);

        assert_eq!(result.len(), 2);
        // First ROC: (105 - 100) / 100 * 100 = 5%
        assert!((result[0] - 5.0).abs() < 1e-10);
    }
}
