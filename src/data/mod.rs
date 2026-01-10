// Data pipeline for real-time market data

pub mod buffer;
pub mod kraken;
pub mod polymarket;
pub mod types;

pub use buffer::FeatureBuffer;
pub use kraken::KrakenClient;
pub use polymarket::PolymarketClient;
pub use types::{
    Asset, AssetFeatures, Candle, MarketSnapshot, OrderBook, PolymarketData, Trade, TradeFlow,
    TradeSide,
};
