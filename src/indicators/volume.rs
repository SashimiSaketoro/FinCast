// Volume indicators

use super::core::sma;

/// On-Balance Volume (OBV)
pub fn obv(close: &[f64], volume: &[f64]) -> Vec<f64> {
    if close.len() < 2 || volume.len() < 2 {
        return vec![];
    }

    let mut result = Vec::with_capacity(close.len());
    result.push(volume[0]);

    for i in 1..close.len() {
        let prev_obv = result[i - 1];

        if close[i] > close[i - 1] {
            result.push(prev_obv + volume[i]);
        } else if close[i] < close[i - 1] {
            result.push(prev_obv - volume[i]);
        } else {
            result.push(prev_obv);
        }
    }

    result
}

/// Volume Rate of Change
pub fn volume_roc(volume: &[f64], period: usize) -> Vec<f64> {
    if volume.len() < period + 1 {
        return vec![];
    }

    volume
        .windows(period + 1)
        .map(|w| {
            let old = w[0];
            let new = w[period];
            if old == 0.0 {
                0.0
            } else {
                ((new - old) / old) * 100.0
            }
        })
        .collect()
}

/// Volume-Weighted Average Price (VWAP)
pub fn vwap(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
    if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
        return vec![];
    }

    let mut cumulative_tpv = 0.0; // Typical price * volume
    let mut cumulative_volume = 0.0;
    let mut result = Vec::with_capacity(close.len());

    for i in 0..close.len() {
        let typical_price = (high[i] + low[i] + close[i]) / 3.0;
        cumulative_tpv += typical_price * volume[i];
        cumulative_volume += volume[i];

        if cumulative_volume == 0.0 {
            result.push(typical_price);
        } else {
            result.push(cumulative_tpv / cumulative_volume);
        }
    }

    result
}

/// Money Flow Index (MFI)
pub fn mfi(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], period: usize) -> Vec<f64> {
    if high.len() < period + 1 || close.len() < period + 1 {
        return vec![];
    }

    // Calculate typical prices
    let mut typical_prices = Vec::new();
    for i in 0..close.len() {
        typical_prices.push((high[i] + low[i] + close[i]) / 3.0);
    }

    // Calculate money flow
    let mut positive_mf = Vec::new();
    let mut negative_mf = Vec::new();

    positive_mf.push(0.0);
    negative_mf.push(0.0);

    for i in 1..typical_prices.len() {
        let money_flow = typical_prices[i] * volume[i];

        if typical_prices[i] > typical_prices[i - 1] {
            positive_mf.push(money_flow);
            negative_mf.push(0.0);
        } else if typical_prices[i] < typical_prices[i - 1] {
            positive_mf.push(0.0);
            negative_mf.push(money_flow);
        } else {
            positive_mf.push(0.0);
            negative_mf.push(0.0);
        }
    }

    // Calculate MFI
    let mut result = Vec::new();

    for i in period..positive_mf.len() {
        let start_idx = i.saturating_sub(period - 1);
        let pos_sum: f64 = positive_mf[start_idx..=i].iter().sum();
        let neg_sum: f64 = negative_mf[start_idx..=i].iter().sum();

        let mfi_val = if neg_sum == 0.0 {
            100.0
        } else {
            let money_ratio = pos_sum / neg_sum;
            100.0 - (100.0 / (1.0 + money_ratio))
        };

        result.push(mfi_val);
    }

    result
}

/// Accumulation/Distribution Line (A/D Line)
pub fn accumulation_distribution(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
) -> Vec<f64> {
    if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
        return vec![];
    }

    let mut result = Vec::with_capacity(close.len());
    let mut ad = 0.0;

    for i in 0..close.len() {
        let clv = if high[i] == low[i] {
            0.0
        } else {
            ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
        };

        ad += clv * volume[i];
        result.push(ad);
    }

    result
}

/// Chaikin Money Flow (CMF)
pub fn chaikin_money_flow(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    period: usize,
) -> Vec<f64> {
    if high.len() < period {
        return vec![];
    }

    let mut result = Vec::new();

    for i in period - 1..close.len() {
        let mut mfv_sum = 0.0;
        let mut vol_sum = 0.0;

        let start_idx = i.saturating_sub(period - 1);
        for j in start_idx..=i {
            let clv = if high[j] == low[j] {
                0.0
            } else {
                ((close[j] - low[j]) - (high[j] - close[j])) / (high[j] - low[j])
            };

            mfv_sum += clv * volume[j];
            vol_sum += volume[j];
        }

        if vol_sum == 0.0 {
            result.push(0.0);
        } else {
            result.push(mfv_sum / vol_sum);
        }
    }

    result
}

/// Volume-weighted Moving Average
pub fn vwma(close: &[f64], volume: &[f64], period: usize) -> Vec<f64> {
    if close.len() < period || volume.len() < period {
        return vec![];
    }

    let mut result = Vec::new();

    for i in period - 1..close.len() {
        let mut weighted_sum = 0.0;
        let mut vol_sum = 0.0;

        let start_idx = i.saturating_sub(period - 1);
        for j in start_idx..=i {
            weighted_sum += close[j] * volume[j];
            vol_sum += volume[j];
        }

        if vol_sum == 0.0 {
            result.push(close[i]);
        } else {
            result.push(weighted_sum / vol_sum);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obv() {
        let close = vec![10.0, 11.0, 10.5, 11.5, 12.0];
        let volume = vec![100.0, 150.0, 120.0, 180.0, 200.0];

        let result = obv(&close, &volume);

        assert_eq!(result.len(), 5);
        assert_eq!(result[0], 100.0);
        assert_eq!(result[1], 250.0); // Price up: 100 + 150
    }

    #[test]
    fn test_vwap() {
        let high = vec![11.0, 12.0, 11.5];
        let low = vec![9.0, 10.0, 9.5];
        let close = vec![10.0, 11.0, 10.5];
        let volume = vec![100.0, 150.0, 120.0];

        let result = vwap(&high, &low, &close, &volume);

        assert_eq!(result.len(), 3);
        for &val in &result {
            assert!(val > 0.0);
        }
    }

    #[test]
    fn test_mfi() {
        let high = vec![50.0, 52.0, 51.0, 53.0, 52.5, 54.0];
        let low = vec![48.0, 49.0, 48.5, 50.0, 49.5, 51.0];
        let close = vec![49.0, 51.0, 50.0, 52.0, 51.0, 53.0];
        let volume = vec![100.0, 150.0, 120.0, 180.0, 160.0, 200.0];

        let result = mfi(&high, &low, &close, &volume, 4);

        assert!(!result.is_empty());
        for &val in &result {
            assert!(val >= 0.0 && val <= 100.0);
        }
    }
}
