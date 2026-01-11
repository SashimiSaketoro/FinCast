// Fibonacci-based indicators

/// Fibonacci levels
#[derive(Debug, Clone)]
pub struct FibonacciLevels {
    pub high: f64,
    pub low: f64,
    pub range: f64,

    // Retracement levels (from high to low)
    pub ret_0: f64,      // 0% (high)
    pub ret_236: f64,    // 23.6%
    pub ret_382: f64,    // 38.2%
    pub ret_500: f64,    // 50%
    pub ret_618: f64,    // 61.8% (golden ratio)
    pub ret_786: f64,    // 78.6%
    pub ret_100: f64,    // 100% (low)

    // Extension levels (beyond high/low)
    pub ext_1272: f64,   // 127.2%
    pub ext_1618: f64,   // 161.8%
    pub ext_2618: f64,   // 261.8%
    pub ext_4236: f64,   // 423.6%
}

/// Calculate Fibonacci retracement and extension levels
pub fn fibonacci_levels(high: f64, low: f64) -> FibonacciLevels {
    let range = high - low;

    FibonacciLevels {
        high,
        low,
        range,

        // Retracements (from high going down)
        ret_0: high,
        ret_236: high - 0.236 * range,
        ret_382: high - 0.382 * range,
        ret_500: high - 0.500 * range,
        ret_618: high - 0.618 * range,
        ret_786: high - 0.786 * range,
        ret_100: low,

        // Extensions (beyond the range)
        ext_1272: high + 0.272 * range,
        ext_1618: high + 0.618 * range,
        ext_2618: high + 1.618 * range,
        ext_4236: high + 3.236 * range,
    }
}

/// Calculate Fibonacci levels from price data over a period
pub fn fibonacci_from_period(data: &[f64], period: usize) -> Vec<FibonacciLevels> {
    if data.len() < period {
        return vec![];
    }

    let mut result = Vec::new();

    for i in period - 1..data.len() {
        let start_idx = i.saturating_sub(period - 1);
        let window = &data[start_idx..=i];
        let high = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let low = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        result.push(fibonacci_levels(high, low));
    }

    result
}

/// Calculate distance to nearest Fibonacci level
/// Returns value in [-1, 1] where 0 is at a fib level
pub fn distance_to_fib_level(price: f64, fib: &FibonacciLevels) -> f64 {
    let levels = vec![
        fib.ret_0, fib.ret_236, fib.ret_382, fib.ret_500,
        fib.ret_618, fib.ret_786, fib.ret_100,
        fib.ext_1272, fib.ext_1618,
    ];

    // Find nearest level
    let mut min_distance = f64::INFINITY;
    let mut nearest_level = fib.ret_500;

    for &level in &levels {
        let dist = (price - level).abs();
        if dist < min_distance {
            min_distance = dist;
            nearest_level = level;
        }
    }

    // Normalize by range
    if fib.range == 0.0 {
        0.0
    } else {
        let normalized = (price - nearest_level) / fib.range;
        normalized.max(-1.0).min(1.0)
    }
}

/// Calculate Fibonacci pivot points
#[derive(Debug, Clone)]
pub struct FibonacciPivots {
    pub pivot: f64,
    pub r1: f64,
    pub r2: f64,
    pub r3: f64,
    pub s1: f64,
    pub s2: f64,
    pub s3: f64,
}

/// Calculate Fibonacci pivot points from previous day's OHLC
pub fn fibonacci_pivots(high: f64, low: f64, close: f64) -> FibonacciPivots {
    let pivot = (high + low + close) / 3.0;
    let range = high - low;

    FibonacciPivots {
        pivot,
        r1: pivot + 0.382 * range,
        r2: pivot + 0.618 * range,
        r3: pivot + 1.000 * range,
        s1: pivot - 0.382 * range,
        s2: pivot - 0.618 * range,
        s3: pivot - 1.000 * range,
    }
}

/// Calculate rolling Fibonacci pivots
pub fn rolling_fibonacci_pivots(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Vec<FibonacciPivots> {
    if high.len() < period || low.len() < period || close.len() < period {
        return vec![];
    }

    let mut result = Vec::new();

    for i in period - 1..close.len() {
        let start_idx = i.saturating_sub(period - 1);
        let h = high[start_idx..=i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let l = low[start_idx..=i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let c = close[i];

        result.push(fibonacci_pivots(h, l, c));
    }

    result
}

/// Fibonacci time zones (for time-based analysis)
/// Returns Fibonacci sequence indices: 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144...
pub fn fibonacci_time_zones(max_index: usize) -> Vec<usize> {
    let mut result = vec![1, 2];

    while result.len() < max_index {
        let next = result[result.len() - 1] + result[result.len() - 2];
        if next > max_index {
            break;
        }
        result.push(next);
    }

    result
}

/// Calculate Fibonacci fan angles (support/resistance lines)
/// Returns slopes based on Fibonacci ratios
pub fn fibonacci_fan_slopes(start_price: f64, end_price: f64, periods: usize) -> Vec<f64> {
    let price_change = end_price - start_price;
    let periods_f64 = periods as f64;

    vec![
        price_change * 0.382 / periods_f64,  // 38.2% line
        price_change * 0.500 / periods_f64,  // 50% line
        price_change * 0.618 / periods_f64,  // 61.8% line
    ]
}

/// Golden ratio (Phi)
pub const PHI: f64 = 1.618033988749895;

/// Calculate if price is near a Fibonacci level (within tolerance)
pub fn is_near_fib_level(price: f64, fib: &FibonacciLevels, tolerance_pct: f64) -> Option<&str> {
    let tolerance = fib.range * tolerance_pct / 100.0;

    let levels = vec![
        (fib.ret_0, "0% (High)"),
        (fib.ret_236, "23.6%"),
        (fib.ret_382, "38.2%"),
        (fib.ret_500, "50%"),
        (fib.ret_618, "61.8% (Golden)"),
        (fib.ret_786, "78.6%"),
        (fib.ret_100, "100% (Low)"),
        (fib.ext_1272, "127.2%"),
        (fib.ext_1618, "161.8%"),
    ];

    for (level, name) in levels {
        if (price - level).abs() <= tolerance {
            return Some(name);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci_levels() {
        let fib = fibonacci_levels(100.0, 80.0);

        assert_eq!(fib.high, 100.0);
        assert_eq!(fib.low, 80.0);
        assert_eq!(fib.range, 20.0);

        // Check 61.8% retracement (golden ratio)
        assert!((fib.ret_618 - 87.64).abs() < 0.01);

        // Check 50% retracement
        assert_eq!(fib.ret_500, 90.0);
    }

    #[test]
    fn test_fibonacci_from_period() {
        let data = vec![100.0, 105.0, 103.0, 110.0, 108.0, 115.0];
        let result = fibonacci_from_period(&data, 3);

        assert_eq!(result.len(), 4);

        // First fib should be from [100, 105, 103]
        assert!(result[0].high >= 103.0);
        assert!(result[0].low <= 100.0);
    }

    #[test]
    fn test_fibonacci_pivots() {
        let pivots = fibonacci_pivots(110.0, 90.0, 100.0);

        assert_eq!(pivots.pivot, 100.0); // (110 + 90 + 100) / 3

        let range = 20.0;
        assert!((pivots.r1 - (100.0 + 0.382 * range)).abs() < 0.01);
        assert!((pivots.s1 - (100.0 - 0.382 * range)).abs() < 0.01);
    }

    #[test]
    fn test_golden_ratio() {
        assert!((PHI - 1.618033988749895).abs() < 1e-10);
    }

    #[test]
    fn test_is_near_fib_level() {
        let fib = fibonacci_levels(100.0, 80.0);

        // 61.8% level is at ~87.64
        assert!(is_near_fib_level(87.6, &fib, 1.0).is_some());
        assert!(is_near_fib_level(50.0, &fib, 1.0).is_none());
    }

    #[test]
    fn test_fibonacci_time_zones() {
        let zones = fibonacci_time_zones(150);

        // Should start with 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144
        assert_eq!(zones[0], 1);
        assert_eq!(zones[1], 2);
        assert_eq!(zones[2], 3);
        assert_eq!(zones[3], 5);
        assert_eq!(zones[4], 8);
        assert_eq!(zones[10], 144);
    }
}
