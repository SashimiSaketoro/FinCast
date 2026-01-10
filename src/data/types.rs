// Common data types for the data pipeline

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Supported cryptocurrency assets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Asset {
    BTC,
    ETH,
    XRP,
    SOL,
    /// Gold (XAU)
    Gold,
}

impl Asset {
    /// Get Kraken trading pair symbol
    pub fn kraken_symbol(&self) -> &'static str {
        match self {
            Asset::BTC => "XBT/USD",
            Asset::ETH => "ETH/USD",
            Asset::XRP => "XRP/USD",
            Asset::SOL => "SOL/USD",
            Asset::Gold => "XAU/USD",
        }
    }

    /// Get all supported assets
    pub fn all() -> Vec<Asset> {
        vec![Asset::BTC, Asset::ETH, Asset::XRP, Asset::SOL, Asset::Gold]
    }

    /// Convert to index for graph structure
    pub fn to_index(&self) -> usize {
        match self {
            Asset::BTC => 0,
            Asset::ETH => 1,
            Asset::XRP => 2,
            Asset::SOL => 3,
            Asset::Gold => 4,
        }
    }

    /// Get from index
    pub fn from_index(idx: usize) -> Option<Asset> {
        match idx {
            0 => Some(Asset::BTC),
            1 => Some(Asset::ETH),
            2 => Some(Asset::XRP),
            3 => Some(Asset::SOL),
            4 => Some(Asset::Gold),
            _ => None,
        }
    }
}

/// OHLCV candle data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub timestamp: DateTime<Utc>,
    pub bids: Vec<(f64, f64)>, // (price, volume)
    pub asks: Vec<(f64, f64)>,
}

impl OrderBook {
    /// Calculate bid-ask spread
    pub fn spread(&self) -> Option<f64> {
        let best_bid = self.bids.first()?.0;
        let best_ask = self.asks.first()?.0;
        Some(best_ask - best_bid)
    }

    /// Calculate spread percentage
    pub fn spread_pct(&self) -> Option<f64> {
        let spread = self.spread()?;
        let mid = self.mid_price()?;
        Some(spread / mid * 100.0)
    }

    /// Calculate mid price
    pub fn mid_price(&self) -> Option<f64> {
        let best_bid = self.bids.first()?.0;
        let best_ask = self.asks.first()?.0;
        Some((best_bid + best_ask) / 2.0)
    }

    /// Calculate order book imbalance
    /// Positive = more buy pressure, Negative = more sell pressure
    pub fn imbalance(&self, depth: usize) -> f64 {
        let bid_volume: f64 = self.bids.iter().take(depth).map(|(_, v)| v).sum();
        let ask_volume: f64 = self.asks.iter().take(depth).map(|(_, v)| v).sum();

        if bid_volume + ask_volume == 0.0 {
            0.0
        } else {
            (bid_volume - ask_volume) / (bid_volume + ask_volume)
        }
    }

    /// Calculate weighted mid price (volume-weighted)
    pub fn weighted_mid_price(&self, depth: usize) -> Option<f64> {
        let mut bid_sum = 0.0;
        let mut bid_vol = 0.0;
        let mut ask_sum = 0.0;
        let mut ask_vol = 0.0;

        for (price, volume) in self.bids.iter().take(depth) {
            bid_sum += price * volume;
            bid_vol += volume;
        }

        for (price, volume) in self.asks.iter().take(depth) {
            ask_sum += price * volume;
            ask_vol += volume;
        }

        if bid_vol + ask_vol == 0.0 {
            return None;
        }

        Some((bid_sum + ask_sum) / (bid_vol + ask_vol))
    }
}

/// Trade data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub timestamp: DateTime<Utc>,
    pub price: f64,
    pub volume: f64,
    pub side: TradeSide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Aggregated trade flow metrics
#[derive(Debug, Clone, Default)]
pub struct TradeFlow {
    pub timestamp: DateTime<Utc>,
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub trade_count: usize,
    pub vwap: f64, // Volume-weighted average price
}

impl TradeFlow {
    /// Calculate buy/sell ratio
    pub fn buy_sell_ratio(&self) -> f64 {
        if self.sell_volume == 0.0 {
            return f64::INFINITY;
        }
        self.buy_volume / self.sell_volume
    }

    /// Calculate volume imbalance
    pub fn volume_imbalance(&self) -> f64 {
        let total = self.buy_volume + self.sell_volume;
        if total == 0.0 {
            0.0
        } else {
            (self.buy_volume - self.sell_volume) / total
        }
    }
}

/// Polymarket futures data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolymarketData {
    pub timestamp: DateTime<Utc>,
    pub asset: Asset,
    pub futures_price: Option<f64>,
    pub open_interest: Option<f64>,
    pub funding_rate: Option<f64>,
    pub sentiment_score: Option<f64>,
}

/// Complete market snapshot for one asset
#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    pub asset: Asset,
    pub timestamp: DateTime<Utc>,
    pub candle: Option<Candle>,
    pub orderbook: Option<OrderBook>,
    pub trade_flow: Option<TradeFlow>,
    pub polymarket: Option<PolymarketData>,
}

impl MarketSnapshot {
    pub fn new(asset: Asset, timestamp: DateTime<Utc>) -> Self {
        Self {
            asset,
            timestamp,
            candle: None,
            orderbook: None,
            trade_flow: None,
            polymarket: None,
        }
    }
}

/// Feature vector for one asset at one timestamp
/// This will be expanded to 128 dimensions with indicators
#[derive(Debug, Clone)]
pub struct AssetFeatures {
    pub asset: Asset,
    pub timestamp: DateTime<Utc>,

    // Raw market features (~30 dims)
    pub price_features: Vec<f64>,      // OHLCV, returns, volatility
    pub orderbook_features: Vec<f64>,   // spreads, depths, imbalances
    pub flow_features: Vec<f64>,        // trade flow, volume metrics

    // Will be computed later:
    // pub indicator_features: Vec<f64>,   // Technical indicators (~70 dims)
    // pub polymarket_features: Vec<f64>,  // Futures/sentiment (~18 dims)
    // pub cross_asset_features: Vec<f64>, // Correlations (~10 dims)
}

impl AssetFeatures {
    /// Total feature dimension (will be 128)
    pub const TOTAL_DIM: usize = 128;

    /// Create empty feature vector
    pub fn empty(asset: Asset, timestamp: DateTime<Utc>) -> Self {
        Self {
            asset,
            timestamp,
            price_features: vec![],
            orderbook_features: vec![],
            flow_features: vec![],
        }
    }

    /// Convert to flat f32 array for model input
    pub fn to_array(&self) -> Vec<f32> {
        let mut features = Vec::with_capacity(Self::TOTAL_DIM);

        // Concatenate all feature groups
        features.extend(self.price_features.iter().map(|&x| x as f32));
        features.extend(self.orderbook_features.iter().map(|&x| x as f32));
        features.extend(self.flow_features.iter().map(|&x| x as f32));

        // Pad to 128 dimensions if needed
        while features.len() < Self::TOTAL_DIM {
            features.push(0.0);
        }

        // Truncate if too long
        features.truncate(Self::TOTAL_DIM);

        features
    }
}
