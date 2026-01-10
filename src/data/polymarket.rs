// Polymarket API client for futures and sentiment data
//
// Polymarket API: https://docs.polymarket.com/

use anyhow::{Context, Result};
use chrono::Utc;
use log::{debug, error, info};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{interval, Duration};

use super::types::{Asset, PolymarketData};

const POLYMARKET_API_BASE: &str = "https://clob.polymarket.com";
const GAMMA_API_BASE: &str = "https://gamma-api.polymarket.com";

/// Polymarket API client
pub struct PolymarketClient {
    client: Client,
    assets: Vec<Asset>,
    latest_data: Arc<RwLock<HashMap<Asset, PolymarketData>>>,
}

#[derive(Debug, Deserialize)]
struct Market {
    #[serde(rename = "condition_id")]
    condition_id: String,
    question: String,
    #[serde(rename = "end_date_iso")]
    end_date: Option<String>,
    tokens: Vec<Token>,
}

#[derive(Debug, Deserialize)]
struct Token {
    token_id: String,
    outcome: String,
    price: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OrderbookResponse {
    bids: Vec<Order>,
    asks: Vec<Order>,
}

#[derive(Debug, Deserialize)]
struct Order {
    price: String,
    size: String,
}

impl PolymarketClient {
    /// Create a new Polymarket client
    pub fn new(assets: Vec<Asset>) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .expect("Failed to create HTTP client"),
            assets,
            latest_data: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start polling Polymarket data at regular intervals
    pub async fn run(&self, poll_interval_secs: u64) -> Result<()> {
        info!("Starting Polymarket client with {}s interval", poll_interval_secs);

        let mut ticker = interval(Duration::from_secs(poll_interval_secs));

        loop {
            ticker.tick().await;

            for asset in &self.assets {
                match self.fetch_asset_data(*asset).await {
                    Ok(data) => {
                        debug!("Fetched Polymarket data for {:?}", asset);
                        let mut latest = self.latest_data.write().await;
                        latest.insert(*asset, data);
                    }
                    Err(e) => {
                        error!("Failed to fetch Polymarket data for {:?}: {}", asset, e);
                    }
                }
            }
        }
    }

    /// Fetch data for a specific asset
    async fn fetch_asset_data(&self, asset: Asset) -> Result<PolymarketData> {
        let search_query = self.asset_search_query(asset);

        // Search for markets related to this asset
        let markets = self.search_markets(&search_query).await?;

        // For now, we'll construct basic data
        // In production, you'd parse specific markets for price predictions, etc.
        let mut data = PolymarketData {
            timestamp: Utc::now(),
            asset,
            futures_price: None,
            open_interest: None,
            funding_rate: None,
            sentiment_score: None,
        };

        // Try to extract useful data from markets
        if let Some(market) = markets.first() {
            // Extract price data from tokens
            if let Some(token) = market.tokens.first() {
                if let Some(price_str) = &token.price {
                    if let Ok(price) = price_str.parse::<f64>() {
                        data.futures_price = Some(price);
                    }
                }
            }

            // Calculate sentiment from market prices
            // Higher prices on "bullish" outcomes = positive sentiment
            data.sentiment_score = self.calculate_sentiment(&market);
        }

        Ok(data)
    }

    /// Search for markets related to an asset
    async fn search_markets(&self, query: &str) -> Result<Vec<Market>> {
        let url = format!("{}/markets?query={}", GAMMA_API_BASE, query);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to search markets")?;

        if !response.status().is_success() {
            anyhow::bail!("API request failed with status: {}", response.status());
        }

        let markets: Vec<Market> = response
            .json()
            .await
            .context("Failed to parse markets response")?;

        Ok(markets)
    }

    /// Get search query for an asset
    fn asset_search_query(&self, asset: Asset) -> String {
        match asset {
            Asset::BTC => "Bitcoin price",
            Asset::ETH => "Ethereum price",
            Asset::XRP => "XRP price",
            Asset::SOL => "Solana price",
            Asset::Gold => "Gold price",
        }
        .to_string()
    }

    /// Calculate sentiment score from market data
    /// Returns value in [-1.0, 1.0] where positive = bullish
    fn calculate_sentiment(&self, market: &Market) -> Option<f64> {
        // Look for tokens with bullish/bearish outcomes
        let mut bullish_price = 0.0;
        let mut bearish_price = 0.0;
        let mut found_outcome = false;

        for token in &market.tokens {
            if let Some(price_str) = &token.price {
                if let Ok(price) = price_str.parse::<f64>() {
                    let outcome_lower = token.outcome.to_lowercase();

                    if outcome_lower.contains("up")
                        || outcome_lower.contains("bull")
                        || outcome_lower.contains("higher")
                        || outcome_lower.contains("yes")
                    {
                        bullish_price = price;
                        found_outcome = true;
                    } else if outcome_lower.contains("down")
                        || outcome_lower.contains("bear")
                        || outcome_lower.contains("lower")
                        || outcome_lower.contains("no")
                    {
                        bearish_price = price;
                        found_outcome = true;
                    }
                }
            }
        }

        if found_outcome {
            // Sentiment = (bullish - bearish) / (bullish + bearish)
            let total = bullish_price + bearish_price;
            if total > 0.0 {
                Some((bullish_price - bearish_price) / total)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Get latest data for an asset
    pub async fn get_latest(&self, asset: Asset) -> Option<PolymarketData> {
        let latest = self.latest_data.read().await;
        latest.get(&asset).cloned()
    }

    /// Get all latest data
    pub async fn get_all_latest(&self) -> HashMap<Asset, PolymarketData> {
        let latest = self.latest_data.read().await;
        latest.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Run manually: cargo test --lib polymarket_client -- --ignored --nocapture
    async fn test_polymarket_client() {
        env_logger::init();

        let assets = vec![Asset::BTC, Asset::ETH];
        let client = PolymarketClient::new(assets);

        // Fetch data once
        for asset in [Asset::BTC, Asset::ETH] {
            match client.fetch_asset_data(asset).await {
                Ok(data) => {
                    info!("Polymarket data for {:?}: {:?}", asset, data);
                }
                Err(e) => {
                    error!("Failed to fetch data for {:?}: {}", asset, e);
                }
            }
        }
    }

    #[tokio::test]
    #[ignore]
    async fn test_polymarket_stream() {
        env_logger::init();

        let assets = vec![Asset::BTC];
        let client = Arc::new(PolymarketClient::new(assets));

        // Spawn client in background
        let client_clone = client.clone();
        tokio::spawn(async move {
            client_clone.run(60).await.unwrap();
        });

        // Wait and check data
        tokio::time::sleep(Duration::from_secs(65)).await;

        let data = client.get_all_latest().await;
        info!("Latest Polymarket data: {:?}", data);
    }
}
