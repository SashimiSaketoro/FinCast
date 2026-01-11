// Technical indicators calculated in Rust for maximum performance
//
// Focuses on reliable indicators across multiple timescales rather than
// breadth of indicator types. All calculations use efficient rolling windows.

pub mod core;
pub mod momentum;
pub mod trend;
pub mod volatility;
pub mod volume;
pub mod fibonacci;
pub mod engine;

pub use core::{sma, ema, std_dev};
pub use momentum::{rsi, stochastic};
pub use trend::{macd, adx, MacdResult};
pub use volatility::{bollinger_bands, atr, BollingerBands};
pub use volume::{obv, volume_roc, mfi};
pub use fibonacci::{fibonacci_levels, fibonacci_from_period, distance_to_fib_level, FibonacciLevels};
pub use engine::{IndicatorEngine, IndicatorSet, TimescaleConfig};
