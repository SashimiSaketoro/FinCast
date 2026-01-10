// Kraken WebSocket client for real-time market data
//
// Uses Kraken WebSocket API v2: https://docs.kraken.com/websockets-v2/

use anyhow::{Context, Result};
use chrono::Utc;
use futures_util::{SinkExt, StreamExt};
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio::time::{sleep, Duration};
use tokio_tungstenite::{connect_async, tungstenite::Message};

use super::types::{Asset, Candle, OrderBook, Trade, TradeSide};

const KRAKEN_WS_URL: &str = "wss://ws.kraken.com/v2";
const RECONNECT_DELAY_SECS: u64 = 5;

/// Kraken WebSocket client
pub struct KrakenClient {
    assets: Vec<Asset>,
    candle_tx: mpsc::UnboundedSender<(Asset, Candle)>,
    orderbook_tx: mpsc::UnboundedSender<(Asset, OrderBook)>,
    trade_tx: mpsc::UnboundedSender<(Asset, Trade)>,
    // Track latest data for each asset
    latest_data: Arc<RwLock<HashMap<Asset, LatestData>>>,
}

#[derive(Debug, Clone, Default)]
struct LatestData {
    candle: Option<Candle>,
    orderbook: Option<OrderBook>,
}

impl KrakenClient {
    /// Create a new Kraken WebSocket client
    pub fn new(
        assets: Vec<Asset>,
    ) -> (
        Self,
        mpsc::UnboundedReceiver<(Asset, Candle)>,
        mpsc::UnboundedReceiver<(Asset, OrderBook)>,
        mpsc::UnboundedReceiver<(Asset, Trade)>,
    ) {
        let (candle_tx, candle_rx) = mpsc::unbounded_channel();
        let (orderbook_tx, orderbook_rx) = mpsc::unbounded_channel();
        let (trade_tx, trade_rx) = mpsc::unbounded_channel();

        let client = Self {
            assets,
            candle_tx,
            orderbook_tx,
            trade_tx,
            latest_data: Arc::new(RwLock::new(HashMap::new())),
        };

        (client, candle_rx, orderbook_rx, trade_rx)
    }

    /// Start the WebSocket connection with automatic reconnection
    pub async fn run(&self) -> Result<()> {
        info!("Starting Kraken WebSocket client");

        loop {
            match self.connect_and_stream().await {
                Ok(_) => {
                    warn!("WebSocket connection closed normally, reconnecting...");
                }
                Err(e) => {
                    error!("WebSocket error: {:?}, reconnecting in {}s", e, RECONNECT_DELAY_SECS);
                }
            }

            sleep(Duration::from_secs(RECONNECT_DELAY_SECS)).await;
        }
    }

    /// Connect to WebSocket and handle streaming
    async fn connect_and_stream(&self) -> Result<()> {
        info!("Connecting to Kraken WebSocket at {}", KRAKEN_WS_URL);

        let (ws_stream, _) = connect_async(KRAKEN_WS_URL)
            .await
            .context("Failed to connect to Kraken WebSocket")?;

        info!("Connected to Kraken WebSocket");

        let (mut write, mut read) = ws_stream.split();

        // Subscribe to channels for all assets
        for asset in &self.assets {
            let symbol = asset.kraken_symbol();

            // Subscribe to OHLC (1-minute candles)
            let ohlc_sub = json!({
                "method": "subscribe",
                "params": {
                    "channel": "ohlc",
                    "symbol": [symbol],
                    "interval": 1
                }
            });

            write
                .send(Message::Text(ohlc_sub.to_string()))
                .await
                .context("Failed to send OHLC subscription")?;

            info!("Subscribed to OHLC for {}", symbol);

            // Subscribe to order book (depth 10)
            let book_sub = json!({
                "method": "subscribe",
                "params": {
                    "channel": "book",
                    "symbol": [symbol],
                    "depth": 10
                }
            });

            write
                .send(Message::Text(book_sub.to_string()))
                .await
                .context("Failed to send book subscription")?;

            info!("Subscribed to order book for {}", symbol);

            // Subscribe to trades
            let trade_sub = json!({
                "method": "subscribe",
                "params": {
                    "channel": "trade",
                    "symbol": [symbol]
                }
            });

            write
                .send(Message::Text(trade_sub.to_string()))
                .await
                .context("Failed to send trade subscription")?;

            info!("Subscribed to trades for {}", symbol);
        }

        // Process incoming messages
        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Err(e) = self.handle_message(&text).await {
                        error!("Error handling message: {:?}", e);
                    }
                }
                Ok(Message::Close(_)) => {
                    warn!("WebSocket closed by server");
                    break;
                }
                Ok(Message::Ping(data)) => {
                    write.send(Message::Pong(data)).await?;
                }
                Err(e) => {
                    error!("WebSocket error: {:?}", e);
                    break;
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Handle incoming WebSocket message
    async fn handle_message(&self, text: &str) -> Result<()> {
        let value: serde_json::Value = serde_json::from_str(text)?;

        // Check message type
        if let Some(channel) = value.get("channel").and_then(|c| c.as_str()) {
            match channel {
                "ohlc" => self.handle_ohlc(&value).await?,
                "book" => self.handle_book(&value).await?,
                "trade" => self.handle_trade(&value).await?,
                _ => debug!("Unhandled channel: {}", channel),
            }
        } else if value.get("method").is_some() {
            // Subscription acknowledgement or status message
            debug!("Status message: {}", text);
        }

        Ok(())
    }

    /// Handle OHLC (candle) data
    async fn handle_ohlc(&self, value: &serde_json::Value) -> Result<()> {
        #[derive(Deserialize)]
        struct OhlcData {
            symbol: String,
            #[serde(rename = "ohlc")]
            candles: Vec<OhlcCandle>,
        }

        #[derive(Deserialize)]
        struct OhlcCandle {
            #[serde(rename = "timestamp")]
            time: String,
            open: String,
            high: String,
            low: String,
            close: String,
            volume: String,
        }

        if let Some(data_array) = value.get("data").and_then(|d| d.as_array()) {
            for data_item in data_array {
                let ohlc_data: OhlcData = serde_json::from_value(data_item.clone())?;

                // Find matching asset
                let asset = self
                    .assets
                    .iter()
                    .find(|a| a.kraken_symbol() == ohlc_data.symbol)
                    .copied();

                if let Some(asset) = asset {
                    for candle_data in ohlc_data.candles {
                        let timestamp = candle_data.time.parse::<f64>()? as i64;
                        let candle = Candle {
                            timestamp: chrono::DateTime::from_timestamp(timestamp, 0)
                                .unwrap_or_else(Utc::now),
                            open: candle_data.open.parse()?,
                            high: candle_data.high.parse()?,
                            low: candle_data.low.parse()?,
                            close: candle_data.close.parse()?,
                            volume: candle_data.volume.parse()?,
                        };

                        debug!("OHLC for {:?}: close={}", asset, candle.close);
                        self.candle_tx.send((asset, candle.clone()))?;

                        // Update latest data
                        let mut latest = self.latest_data.write().await;
                        latest.entry(asset).or_default().candle = Some(candle);
                    }
                }
            }
        }

        Ok(())
    }

    /// Handle order book data
    async fn handle_book(&self, value: &serde_json::Value) -> Result<()> {
        #[derive(Deserialize)]
        struct BookData {
            symbol: String,
            #[serde(rename = "bids")]
            bids: Option<Vec<BookLevel>>,
            #[serde(rename = "asks")]
            asks: Option<Vec<BookLevel>>,
            #[serde(rename = "timestamp")]
            time: Option<String>,
        }

        #[derive(Deserialize)]
        struct BookLevel {
            price: f64,
            qty: f64,
        }

        if let Some(data_array) = value.get("data").and_then(|d| d.as_array()) {
            for data_item in data_array {
                let book_data: BookData = serde_json::from_value(data_item.clone())?;

                // Find matching asset
                let asset = self
                    .assets
                    .iter()
                    .find(|a| a.kraken_symbol() == book_data.symbol)
                    .copied();

                if let Some(asset) = asset {
                    let timestamp = if let Some(time_str) = book_data.time {
                        let ts = time_str.parse::<f64>()? as i64;
                        chrono::DateTime::from_timestamp(ts, 0).unwrap_or_else(Utc::now)
                    } else {
                        Utc::now()
                    };

                    let bids = book_data
                        .bids
                        .unwrap_or_default()
                        .into_iter()
                        .map(|b| (b.price, b.qty))
                        .collect();

                    let asks = book_data
                        .asks
                        .unwrap_or_default()
                        .into_iter()
                        .map(|a| (a.price, a.qty))
                        .collect();

                    let orderbook = OrderBook {
                        timestamp,
                        bids,
                        asks,
                    };

                    debug!(
                        "Order book for {:?}: spread={:?}",
                        asset,
                        orderbook.spread()
                    );
                    self.orderbook_tx.send((asset, orderbook.clone()))?;

                    // Update latest data
                    let mut latest = self.latest_data.write().await;
                    latest.entry(asset).or_default().orderbook = Some(orderbook);
                }
            }
        }

        Ok(())
    }

    /// Handle trade data
    async fn handle_trade(&self, value: &serde_json::Value) -> Result<()> {
        #[derive(Deserialize)]
        struct TradeData {
            symbol: String,
            #[serde(rename = "trades")]
            trades: Vec<TradeItem>,
        }

        #[derive(Deserialize)]
        struct TradeItem {
            price: f64,
            qty: f64,
            #[serde(rename = "timestamp")]
            time: String,
            side: String,
        }

        if let Some(data_array) = value.get("data").and_then(|d| d.as_array()) {
            for data_item in data_array {
                let trade_data: TradeData = serde_json::from_value(data_item.clone())?;

                // Find matching asset
                let asset = self
                    .assets
                    .iter()
                    .find(|a| a.kraken_symbol() == trade_data.symbol)
                    .copied();

                if let Some(asset) = asset {
                    for trade_item in trade_data.trades {
                        let timestamp = trade_item.time.parse::<f64>()? as i64;
                        let trade = Trade {
                            timestamp: chrono::DateTime::from_timestamp(timestamp, 0)
                                .unwrap_or_else(Utc::now),
                            price: trade_item.price,
                            volume: trade_item.qty,
                            side: match trade_item.side.as_str() {
                                "buy" => TradeSide::Buy,
                                "sell" => TradeSide::Sell,
                                _ => TradeSide::Buy, // default
                            },
                        };

                        debug!(
                            "Trade for {:?}: price={}, volume={}",
                            asset, trade.price, trade.volume
                        );
                        self.trade_tx.send((asset, trade))?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Get latest data for an asset
    pub async fn get_latest(&self, asset: Asset) -> Option<LatestData> {
        let latest = self.latest_data.read().await;
        latest.get(&asset).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Run manually: cargo test --lib kraken_client -- --ignored --nocapture
    async fn test_kraken_websocket() {
        env_logger::init();

        let assets = vec![Asset::BTC, Asset::ETH];
        let (client, mut candle_rx, mut orderbook_rx, mut trade_rx) =
            KrakenClient::new(assets);

        // Spawn client in background
        tokio::spawn(async move {
            client.run().await.unwrap();
        });

        // Listen for data for 30 seconds
        let timeout = tokio::time::sleep(Duration::from_secs(30));
        tokio::pin!(timeout);

        loop {
            tokio::select! {
                Some((asset, candle)) = candle_rx.recv() => {
                    info!("Received candle for {:?}: close={}", asset, candle.close);
                }
                Some((asset, book)) = orderbook_rx.recv() => {
                    info!("Received orderbook for {:?}: spread={:?}", asset, book.spread());
                }
                Some((asset, trade)) = trade_rx.recv() => {
                    info!("Received trade for {:?}: price={}", asset, trade.price);
                }
                _ = &mut timeout => {
                    info!("Test timeout reached");
                    break;
                }
            }
        }
    }
}
