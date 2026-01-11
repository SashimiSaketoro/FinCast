// Trend indicators

use super::core::{ema, sma};

/// MACD Result
#[derive(Debug, Clone)]
pub struct MacdResult {
    pub macd: Vec<f64>,
    pub signal: Vec<f64>,
    pub histogram: Vec<f64>,
}

/// Moving Average Convergence Divergence (MACD)
pub fn macd(data: &[f64], fast_period: usize, slow_period: usize, signal_period: usize) -> MacdResult {
    let fast_ema = ema(data, fast_period);
    let slow_ema = ema(data, slow_period);

    // Calculate MACD line (fast EMA - slow EMA)
    let mut macd_line = Vec::new();
    let min_len = fast_ema.len().min(slow_ema.len());

    for i in 0..min_len {
        if fast_ema[i].is_nan() || slow_ema[i].is_nan() {
            macd_line.push(f64::NAN);
        } else {
            macd_line.push(fast_ema[i] - slow_ema[i]);
        }
    }

    // Calculate signal line (EMA of MACD)
    let signal_line = ema(&macd_line, signal_period);

    // Calculate histogram (MACD - Signal)
    let mut histogram = Vec::new();
    let min_len = macd_line.len().min(signal_line.len());

    for i in 0..min_len {
        if macd_line[i].is_nan() || signal_line[i].is_nan() {
            histogram.push(f64::NAN);
        } else {
            histogram.push(macd_line[i] - signal_line[i]);
        }
    }

    MacdResult {
        macd: macd_line,
        signal: signal_line,
        histogram,
    }
}

/// Average Directional Index (ADX)
pub fn adx(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Vec<f64> {
    if high.len() < period + 1 || low.len() < period + 1 || close.len() < period + 1 {
        return vec![];
    }

    let mut plus_dm = Vec::new();
    let mut minus_dm = Vec::new();
    let mut tr = Vec::new();

    // Calculate +DM, -DM, and TR
    for i in 1..high.len() {
        let high_diff = high[i] - high[i - 1];
        let low_diff = low[i - 1] - low[i];

        let plus_dm_val = if high_diff > low_diff && high_diff > 0.0 {
            high_diff
        } else {
            0.0
        };

        let minus_dm_val = if low_diff > high_diff && low_diff > 0.0 {
            low_diff
        } else {
            0.0
        };

        let tr_val = (high[i] - low[i])
            .max((high[i] - close[i - 1]).abs())
            .max((low[i] - close[i - 1]).abs());

        plus_dm.push(plus_dm_val);
        minus_dm.push(minus_dm_val);
        tr.push(tr_val);
    }

    // Smooth with Wilder's smoothing
    let mut plus_di = Vec::new();
    let mut minus_di = Vec::new();

    if plus_dm.len() < period {
        return vec![];
    }

    let mut smoothed_plus_dm: f64 = plus_dm[..period].iter().sum::<f64>();
    let mut smoothed_minus_dm: f64 = minus_dm[..period].iter().sum::<f64>();
    let mut smoothed_tr: f64 = tr[..period].iter().sum::<f64>();

    plus_di.push(100.0 * smoothed_plus_dm / smoothed_tr);
    minus_di.push(100.0 * smoothed_minus_dm / smoothed_tr);

    for i in period..plus_dm.len() {
        smoothed_plus_dm = smoothed_plus_dm - smoothed_plus_dm / period as f64 + plus_dm[i];
        smoothed_minus_dm = smoothed_minus_dm - smoothed_minus_dm / period as f64 + minus_dm[i];
        smoothed_tr = smoothed_tr - smoothed_tr / period as f64 + tr[i];

        plus_di.push(100.0 * smoothed_plus_dm / smoothed_tr);
        minus_di.push(100.0 * smoothed_minus_dm / smoothed_tr);
    }

    // Calculate DX and ADX
    let mut dx = Vec::new();
    for i in 0..plus_di.len() {
        let di_sum = plus_di[i] + minus_di[i];
        let di_diff = (plus_di[i] - minus_di[i]).abs();

        if di_sum == 0.0 {
            dx.push(0.0);
        } else {
            dx.push(100.0 * di_diff / di_sum);
        }
    }

    // Smooth DX to get ADX
    if dx.len() < period {
        return vec![];
    }

    let mut adx_values = Vec::new();
    let mut adx_val: f64 = dx[..period].iter().sum::<f64>() / period as f64;
    adx_values.push(adx_val);

    for i in period..dx.len() {
        adx_val = (adx_val * (period - 1) as f64 + dx[i]) / period as f64;
        adx_values.push(adx_val);
    }

    adx_values
}

/// Parabolic SAR
pub fn parabolic_sar(
    high: &[f64],
    low: &[f64],
    acceleration: f64,
    max_acceleration: f64,
) -> Vec<f64> {
    if high.is_empty() || low.is_empty() {
        return vec![];
    }

    let mut result = Vec::with_capacity(high.len());
    let mut sar = low[0];
    let mut ep = high[0];  // Extreme point
    let mut af = acceleration;
    let mut is_bull = true;

    result.push(sar);

    for i in 1..high.len() {
        // Update SAR
        sar = sar + af * (ep - sar);

        // Determine if trend continues
        let mut trend_changed = false;

        if is_bull {
            if low[i] < sar {
                // Trend reversal to bearish
                is_bull = false;
                sar = ep;
                ep = low[i];
                af = acceleration;
                trend_changed = true;
            }
        } else {
            if high[i] > sar {
                // Trend reversal to bullish
                is_bull = true;
                sar = ep;
                ep = high[i];
                af = acceleration;
                trend_changed = true;
            }
        }

        if !trend_changed {
            // Update extreme point and acceleration
            if is_bull {
                if high[i] > ep {
                    ep = high[i];
                    af = (af + acceleration).min(max_acceleration);
                }
            } else {
                if low[i] < ep {
                    ep = low[i];
                    af = (af + acceleration).min(max_acceleration);
                }
            }
        }

        // SAR should not be within the last two bars' range
        if is_bull {
            sar = sar.min(low[i]).min(low[i - 1]);
        } else {
            sar = sar.max(high[i]).max(high[i - 1]);
        }

        result.push(sar);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macd() {
        let data: Vec<f64> = (0..50).map(|x| (x as f64).sin() * 10.0 + 50.0).collect();
        let result = macd(&data, 12, 26, 9);

        assert!(!result.macd.is_empty());
        assert!(!result.signal.is_empty());
        assert!(!result.histogram.is_empty());
    }

    #[test]
    fn test_adx() {
        let high: Vec<f64> = (0..30).map(|x| 50.0 + (x as f64) * 0.5).collect();
        let low: Vec<f64> = (0..30).map(|x| 48.0 + (x as f64) * 0.5).collect();
        let close: Vec<f64> = (0..30).map(|x| 49.0 + (x as f64) * 0.5).collect();

        let result = adx(&high, &low, &close, 14);

        assert!(!result.is_empty());
        for &val in &result {
            assert!(val >= 0.0 && val <= 100.0);
        }
    }
}
