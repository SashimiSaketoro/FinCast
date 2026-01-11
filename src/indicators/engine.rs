// Multi-timescale indicator engine
//
// Calculates reliable indicators at multiple timescales rather than
// computing many different indicator types. This provides historical
// depth across different market regimes.

use super::*;

/// Configuration for indicator timescales
#[derive(Debug, Clone)]
pub struct TimescaleConfig {
    /// Very short term: scalping, microstructure
    pub very_short: Vec<usize>,  // e.g., [5, 10, 14, 20]

    /// Short term: day trading
    pub short: Vec<usize>,       // e.g., [30, 50]

    /// Medium term: swing trading
    pub medium: Vec<usize>,      // e.g., [100, 200]

    /// Long term: position trading
    pub long: Vec<usize>,        // e.g., [500, 1000]
}

impl Default for TimescaleConfig {
    fn default() -> Self {
        Self {
            very_short: vec![5, 10, 14, 20],
            short: vec![30, 50],
            medium: vec![100, 200],
            long: vec![500],
        }
    }
}

impl TimescaleConfig {
    /// Get all periods as a flat list
    pub fn all_periods(&self) -> Vec<usize> {
        let mut periods = Vec::new();
        periods.extend(&self.very_short);
        periods.extend(&self.short);
        periods.extend(&self.medium);
        periods.extend(&self.long);
        periods
    }

    /// Get total number of periods
    pub fn count(&self) -> usize {
        self.very_short.len() + self.short.len() + self.medium.len() + self.long.len()
    }
}

/// Complete set of indicators for one asset
#[derive(Debug, Clone)]
pub struct IndicatorSet {
    /// RSI values at different timescales
    pub rsi: Vec<(usize, f64)>,  // (period, value)

    /// MACD values at different timescales
    pub macd: Vec<(usize, f64)>,  // (period, macd_value)
    pub macd_signal: Vec<(usize, f64)>,
    pub macd_histogram: Vec<(usize, f64)>,

    /// Bollinger Band metrics at different timescales
    pub bb_bandwidth: Vec<(usize, f64)>,  // (period, bandwidth)
    pub bb_percent_b: Vec<(usize, f64)>,  // (period, %B)

    /// ATR at different timescales
    pub atr: Vec<(usize, f64)>,

    /// Moving averages
    pub sma: Vec<(usize, f64)>,
    pub ema: Vec<(usize, f64)>,

    /// Stochastic
    pub stoch_k: Vec<(usize, f64)>,
    pub stoch_d: Vec<(usize, f64)>,

    /// Fibonacci levels (current)
    pub fib_distance: f64,  // Distance to nearest fib level

    /// ADX (trend strength)
    pub adx: Vec<(usize, f64)>,

    /// Volume indicators
    pub volume_roc: Vec<(usize, f64)>,
    pub mfi: Vec<(usize, f64)>,

    /// Returns at different periods
    pub returns: Vec<(usize, f64)>,
}

impl IndicatorSet {
    /// Create empty indicator set
    pub fn empty() -> Self {
        Self {
            rsi: vec![],
            macd: vec![],
            macd_signal: vec![],
            macd_histogram: vec![],
            bb_bandwidth: vec![],
            bb_percent_b: vec![],
            atr: vec![],
            sma: vec![],
            ema: vec![],
            stoch_k: vec![],
            stoch_d: vec![],
            fib_distance: 0.0,
            adx: vec![],
            volume_roc: vec![],
            mfi: vec![],
            returns: vec![],
        }
    }

    /// Convert to flat feature vector
    pub fn to_feature_vec(&self) -> Vec<f32> {
        let mut features = Vec::new();

        // RSI features
        for (_, val) in &self.rsi {
            features.push(*val as f32);
        }

        // MACD features
        for (_, val) in &self.macd {
            features.push(*val as f32);
        }
        for (_, val) in &self.macd_histogram {
            features.push(*val as f32);
        }

        // Bollinger features
        for (_, val) in &self.bb_bandwidth {
            features.push(*val as f32);
        }
        for (_, val) in &self.bb_percent_b {
            features.push(*val as f32);
        }

        // ATR features
        for (_, val) in &self.atr {
            features.push(*val as f32);
        }

        // Moving averages
        for (_, val) in &self.sma {
            features.push(*val as f32);
        }
        for (_, val) in &self.ema {
            features.push(*val as f32);
        }

        // Stochastic
        for (_, val) in &self.stoch_k {
            features.push(*val as f32);
        }

        // Fibonacci
        features.push(self.fib_distance as f32);

        // ADX
        for (_, val) in &self.adx {
            features.push(*val as f32);
        }

        // Volume
        for (_, val) in &self.volume_roc {
            features.push(*val as f32);
        }
        for (_, val) in &self.mfi {
            features.push(*val as f32);
        }

        // Returns
        for (_, val) in &self.returns {
            features.push(*val as f32);
        }

        features
    }

    /// Get total feature count
    pub fn feature_count(&self) -> usize {
        self.rsi.len()
            + self.macd.len()
            + self.macd_histogram.len()
            + self.bb_bandwidth.len()
            + self.bb_percent_b.len()
            + self.atr.len()
            + self.sma.len()
            + self.ema.len()
            + self.stoch_k.len()
            + 1  // fib_distance
            + self.adx.len()
            + self.volume_roc.len()
            + self.mfi.len()
            + self.returns.len()
    }
}

/// Multi-timescale indicator engine
pub struct IndicatorEngine {
    config: TimescaleConfig,
}

impl IndicatorEngine {
    /// Create a new indicator engine
    pub fn new(config: TimescaleConfig) -> Self {
        Self { config }
    }

    /// Create with default timescales
    pub fn default() -> Self {
        Self::new(TimescaleConfig::default())
    }

    /// Calculate all indicators for given OHLCV data
    ///
    /// Returns the most recent indicator values (for current bar)
    pub fn calculate(
        &self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> IndicatorSet {
        let mut indicators = IndicatorSet::empty();

        // Calculate RSI at all timescales
        for &period in self.config.all_periods().iter() {
            if close.len() > period {
                let rsi_vals = rsi(close, period);
                if let Some(&val) = rsi_vals.last() {
                    indicators.rsi.push((period, val));
                }
            }
        }

        // Calculate MACD at selected timescales
        for &period in &[12, 26, 50, 100] {
            if close.len() > period * 2 {
                let macd_result = macd(close, period, period * 2, 9);
                if let (Some(&m), Some(&h)) = (macd_result.macd.last(), macd_result.histogram.last()) {
                    if !m.is_nan() && !h.is_nan() {
                        indicators.macd.push((period, m));
                        indicators.macd_histogram.push((period, h));
                    }
                }
            }
        }

        // Calculate Bollinger Bands at multiple timescales
        for &period in self.config.all_periods().iter() {
            if close.len() > period {
                let bb = bollinger_bands(close, period, 2.0);
                if let (Some(&bw), Some(&pb)) = (bb.bandwidth.last(), bb.percent_b.last()) {
                    indicators.bb_bandwidth.push((period, bw));
                    indicators.bb_percent_b.push((period, pb));
                }
            }
        }

        // Calculate ATR at all timescales
        for &period in self.config.all_periods().iter() {
            if high.len() > period {
                let atr_vals = atr(high, low, close, period);
                if let Some(&val) = atr_vals.last() {
                    indicators.atr.push((period, val));
                }
            }
        }

        // Calculate moving averages
        for &period in self.config.all_periods().iter() {
            if close.len() >= period {
                let sma_vals = sma(close, period);
                if let Some(&val) = sma_vals.last() {
                    indicators.sma.push((period, val));
                }

                let ema_vals = ema(close, period);
                if let Some(&val) = ema_vals.last() {
                    if !val.is_nan() {
                        indicators.ema.push((period, val));
                    }
                }
            }
        }

        // Calculate Stochastic at key periods
        for &period in &[14, 20, 50] {
            if high.len() > period {
                let stoch = stochastic(high, low, close, period, 3);
                if let Some(&(k, d)) = stoch.last() {
                    indicators.stoch_k.push((period, k));
                    indicators.stoch_d.push((period, d));
                }
            }
        }

        // Calculate Fibonacci levels (last 100 bars)
        if close.len() > 100 {
            let fib_levels = fibonacci_from_period(close, 100);
            if let (Some(fib), Some(&current_price)) = (fib_levels.last(), close.last()) {
                indicators.fib_distance = distance_to_fib_level(current_price, fib);
            }
        }

        // Calculate ADX at key periods
        for &period in &[14, 20, 50] {
            if high.len() > period {
                let adx_vals = adx(high, low, close, period);
                if let Some(&val) = adx_vals.last() {
                    indicators.adx.push((period, val));
                }
            }
        }

        // Calculate volume indicators
        if volume.len() > 0 {
            for &period in &[10, 20, 50] {
                if volume.len() > period {
                    let vroc = volume_roc(volume, period);
                    if let Some(&val) = vroc.last() {
                        indicators.volume_roc.push((period, val));
                    }
                }

                if high.len() > period {
                    let mfi_vals = mfi(high, low, close, volume, period);
                    if let Some(&val) = mfi_vals.last() {
                        indicators.mfi.push((period, val));
                    }
                }
            }
        }

        // Calculate returns at different horizons
        for &period in &[1, 5, 10, 20, 50] {
            if close.len() > period {
                let ret = (close[close.len() - 1] - close[close.len() - 1 - period])
                    / close[close.len() - 1 - period];
                indicators.returns.push((period, ret));
            }
        }

        indicators
    }

    /// Get expected feature dimension from this engine
    pub fn feature_dimension(&self) -> usize {
        let n_periods = self.config.count();

        // RSI: all periods
        let rsi_dim = n_periods;

        // MACD: 4 timescales (12, 26, 50, 100)
        let macd_dim = 4 * 2; // macd + histogram

        // Bollinger: all periods (bandwidth + %B)
        let bb_dim = n_periods * 2;

        // ATR: all periods
        let atr_dim = n_periods;

        // Moving averages: all periods (SMA + EMA)
        let ma_dim = n_periods * 2;

        // Stochastic: 3 periods
        let stoch_dim = 3;

        // Fibonacci: 1
        let fib_dim = 1;

        // ADX: 3 periods
        let adx_dim = 3;

        // Volume: 3 periods (ROC + MFI)
        let vol_dim = 3 * 2;

        // Returns: 5 periods
        let ret_dim = 5;

        rsi_dim + macd_dim + bb_dim + atr_dim + ma_dim + stoch_dim + fib_dim + adx_dim + vol_dim + ret_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timescale_config() {
        let config = TimescaleConfig::default();

        assert!(!config.very_short.is_empty());
        assert!(!config.short.is_empty());
        assert!(!config.medium.is_empty());

        let all = config.all_periods();
        assert!(all.len() > 5);
    }

    #[test]
    fn test_indicator_engine() {
        let engine = IndicatorEngine::default();

        // Generate test data
        let len = 1000;
        let close: Vec<f64> = (0..len).map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0).collect();
        let high: Vec<f64> = close.iter().map(|&c| c + 2.0).collect();
        let low: Vec<f64> = close.iter().map(|&c| c - 2.0).collect();
        let open: Vec<f64> = close.clone();
        let volume: Vec<f64> = vec![1000.0; len];

        let indicators = engine.calculate(&open, &high, &low, &close, &volume);

        // Check we got indicators
        assert!(!indicators.rsi.is_empty());
        assert!(!indicators.bb_bandwidth.is_empty());
        assert!(!indicators.atr.is_empty());
        assert!(!indicators.sma.is_empty());

        // Check feature vector
        let features = indicators.to_feature_vec();
        assert!(features.len() > 50);

        println!("Feature dimension: {}", indicators.feature_count());
        println!("Expected dimension: {}", engine.feature_dimension());
    }

    #[test]
    fn test_indicator_set_to_vec() {
        let mut indicators = IndicatorSet::empty();

        indicators.rsi.push((14, 45.5));
        indicators.rsi.push((20, 48.2));
        indicators.bb_bandwidth.push((20, 0.15));

        let features = indicators.to_feature_vec();

        // Features: 2 RSI + 1 BB bandwidth + 1 fib_distance (default 0.0)
        assert_eq!(features.len(), 4);
        assert_eq!(features[0], 45.5);
        assert_eq!(features[1], 48.2);
        assert_eq!(features[2], 0.15);
        assert_eq!(features[3], 0.0);  // fib_distance default
    }
}
