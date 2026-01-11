// Volatility indicators

use super::core::{ema, sma, std_dev};

/// Bollinger Bands
#[derive(Debug, Clone)]
pub struct BollingerBands {
    pub upper: Vec<f64>,
    pub middle: Vec<f64>,
    pub lower: Vec<f64>,
    pub bandwidth: Vec<f64>,
    pub percent_b: Vec<f64>,
}

/// Calculate Bollinger Bands
pub fn bollinger_bands(data: &[f64], period: usize, std_multiplier: f64) -> BollingerBands {
    let middle = sma(data, period);
    let std = std_dev(data, period);

    let mut upper = Vec::new();
    let mut lower = Vec::new();
    let mut bandwidth = Vec::new();
    let mut percent_b = Vec::new();

    for i in 0..middle.len() {
        let u = middle[i] + std_multiplier * std[i];
        let l = middle[i] - std_multiplier * std[i];

        upper.push(u);
        lower.push(l);

        // Bandwidth: (upper - lower) / middle
        if middle[i] != 0.0 {
            bandwidth.push((u - l) / middle[i]);
        } else {
            bandwidth.push(0.0);
        }

        // %B: (price - lower) / (upper - lower)
        let price_idx = i + period - 1;
        if price_idx < data.len() && u != l {
            percent_b.push((data[price_idx] - l) / (u - l));
        } else {
            percent_b.push(0.5); // Neutral
        }
    }

    BollingerBands {
        upper,
        middle,
        lower,
        bandwidth,
        percent_b,
    }
}

/// Average True Range (ATR)
pub fn atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    if high.len() < period + 1 || low.len() < period + 1 || close.len() < period + 1 {
        return vec![];
    }

    let mut true_ranges = Vec::new();

    for i in 1..high.len() {
        let tr1 = high[i] - low[i];
        let tr2 = (high[i] - close[i - 1]).abs();
        let tr3 = (low[i] - close[i - 1]).abs();

        true_ranges.push(tr1.max(tr2).max(tr3));
    }

    // First ATR is SMA
    let mut result = Vec::new();
    let mut atr_val: f64 = true_ranges[..period].iter().sum::<f64>() / period as f64;
    result.push(atr_val);

    // Subsequent ATR uses smoothing (Wilder's)
    for i in period..true_ranges.len() {
        atr_val = (atr_val * (period - 1) as f64 + true_ranges[i]) / period as f64;
        result.push(atr_val);
    }

    result
}

/// Historical Volatility (using standard deviation of log returns)
pub fn historical_volatility(data: &[f64], period: usize, annualization_factor: f64) -> Vec<f64> {
    if data.len() < period + 1 {
        return vec![];
    }

    // Calculate log returns
    let mut log_returns = Vec::new();
    for i in 1..data.len() {
        log_returns.push((data[i] / data[i - 1]).ln());
    }

    // Calculate rolling standard deviation
    let mut result = Vec::new();

    for i in period - 1..log_returns.len() {
        let start_idx = i.saturating_sub(period - 1);
        let window = &log_returns[start_idx..=i];
        let mean = window.iter().sum::<f64>() / period as f64;

        let variance = window
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / period as f64;

        // Annualize the volatility
        result.push(variance.sqrt() * annualization_factor.sqrt());
    }

    result
}

/// Keltner Channels
#[derive(Debug, Clone)]
pub struct KeltnerChannels {
    pub upper: Vec<f64>,
    pub middle: Vec<f64>,
    pub lower: Vec<f64>,
}

/// Calculate Keltner Channels
pub fn keltner_channels(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    ema_period: usize,
    atr_period: usize,
    multiplier: f64,
) -> KeltnerChannels {
    let middle = ema(close, ema_period);
    let atr_vals = atr(high, low, close, atr_period);

    let mut upper = Vec::new();
    let mut lower = Vec::new();

    let min_len = middle.len().min(atr_vals.len());

    for i in 0..min_len {
        if !middle[i].is_nan() {
            upper.push(middle[i] + multiplier * atr_vals[i]);
            lower.push(middle[i] - multiplier * atr_vals[i]);
        }
    }

    KeltnerChannels {
        upper,
        middle: middle[..min_len].to_vec(),
        lower,
    }
}

/// Donchian Channels
#[derive(Debug, Clone)]
pub struct DonchianChannels {
    pub upper: Vec<f64>,
    pub middle: Vec<f64>,
    pub lower: Vec<f64>,
}

/// Calculate Donchian Channels
pub fn donchian_channels(high: &[f64], low: &[f64], period: usize) -> DonchianChannels {
    if high.len() < period || low.len() < period {
        return DonchianChannels {
            upper: vec![],
            middle: vec![],
            lower: vec![],
        };
    }

    let mut upper = Vec::new();
    let mut lower = Vec::new();
    let mut middle = Vec::new();

    for i in period - 1..high.len() {
        let start_idx = i.saturating_sub(period - 1);
        let highest = high[start_idx..=i]
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let lowest = low[start_idx..=i]
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));

        upper.push(highest);
        lower.push(lowest);
        middle.push((highest + lowest) / 2.0);
    }

    DonchianChannels {
        upper,
        middle,
        lower,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bollinger_bands() {
        let data = vec![10.0, 11.0, 12.0, 11.5, 12.5, 13.0, 12.5];
        let bb = bollinger_bands(&data, 5, 2.0);

        assert!(!bb.upper.is_empty());
        assert!(!bb.middle.is_empty());
        assert!(!bb.lower.is_empty());

        // Upper should be above middle, middle above lower
        for i in 0..bb.middle.len() {
            assert!(bb.upper[i] >= bb.middle[i]);
            assert!(bb.middle[i] >= bb.lower[i]);
        }
    }

    #[test]
    fn test_atr() {
        let high = vec![50.0, 52.0, 51.0, 53.0, 52.5];
        let low = vec![48.0, 49.0, 48.5, 50.0, 49.5];
        let close = vec![49.0, 51.0, 50.0, 52.0, 51.0];

        let result = atr(&high, &low, &close, 3);

        assert!(!result.is_empty());
        for &val in &result {
            assert!(val > 0.0); // ATR should be positive
        }
    }

    #[test]
    fn test_historical_volatility() {
        let data: Vec<f64> = (0..50).map(|x| 100.0 + (x as f64) * 0.5).collect();
        let result = historical_volatility(&data, 20, 252.0); // Annualize with 252 trading days

        assert!(!result.is_empty());
        for &val in &result {
            assert!(val >= 0.0);
        }
    }
}
