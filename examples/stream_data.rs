// Example: Stream live market data from Kraken and Polymarket
//
// Run with: cargo run --example stream_data
//
// This demonstrates the real-time data pipeline for STG-Mamba

use anyhow::Result;
use log::info;
use std::sync::Arc;
use std::time::Duration;
use stg_mamba::data::{Asset, FeatureBuffer, KrakenClient, PolymarketClient};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    info!("Starting STG-Mamba data pipeline demo");

    // Define assets to track
    let assets = vec![Asset::BTC, Asset::ETH, Asset::XRP, Asset::SOL, Asset::Gold];

    info!("Tracking assets: {:?}", assets);

    // Create Kraken WebSocket client
    let (kraken_client, mut candle_rx, mut orderbook_rx, mut trade_rx) =
        KrakenClient::new(assets.clone());

    // Create Polymarket API client
    let polymarket_client = Arc::new(PolymarketClient::new(assets.clone()));

    // Create feature buffer (store last 60 minutes at 1-min resolution)
    let mut feature_buffer = FeatureBuffer::new(assets.clone(), 60);

    info!("Starting Kraken WebSocket...");

    // Spawn Kraken client
    let kraken_handle = tokio::spawn(async move {
        kraken_client.run().await.expect("Kraken client failed");
    });

    info!("Starting Polymarket API client...");

    // Spawn Polymarket client (poll every 60 seconds)
    let polymarket_handle = {
        let client = polymarket_client.clone();
        tokio::spawn(async move {
            client.run(60).await.expect("Polymarket client failed");
        })
    };

    info!("Data pipeline running! Receiving live data...\n");

    // Listen for data
    let mut candle_count = 0;
    let mut orderbook_count = 0;
    let mut trade_count = 0;

    let timeout = tokio::time::sleep(Duration::from_secs(300)); // Run for 5 minutes
    tokio::pin!(timeout);

    loop {
        tokio::select! {
            Some((asset, candle)) = candle_rx.recv() => {
                candle_count += 1;
                info!(
                    "📊 CANDLE [{:?}] Close: ${:.2}, Volume: {:.2}",
                    asset, candle.close, candle.volume
                );

                // In production, you'd extract features and push to buffer here
                // feature_buffer.push(features);
            }

            Some((asset, orderbook)) = orderbook_rx.recv() => {
                orderbook_count += 1;
                if let (Some(spread), Some(imb)) = (orderbook.spread_pct(), Some(orderbook.imbalance(10))) {
                    info!(
                        "📖 BOOK [{:?}] Spread: {:.3}%, Imbalance: {:.3}",
                        asset, spread, imb
                    );
                }
            }

            Some((asset, trade)) = trade_rx.recv() => {
                trade_count += 1;
                info!(
                    "💰 TRADE [{:?}] {:?} @ ${:.2}, Volume: {:.4}",
                    asset, trade.side, trade.price, trade.volume
                );
            }

            _ = &mut timeout => {
                info!("\n⏰ Timeout reached (5 minutes)");
                break;
            }
        }
    }

    info!("\n📈 Summary:");
    info!("  Candles received: {}", candle_count);
    info!("  Order books received: {}", orderbook_count);
    info!("  Trades received: {}", trade_count);

    // Check Polymarket data
    let polymarket_data = polymarket_client.get_all_latest().await;
    info!("\n🎲 Polymarket data:");
    for (asset, data) in polymarket_data {
        info!(
            "  {:?}: Sentiment: {:?}, Price: {:?}",
            asset, data.sentiment_score, data.futures_price
        );
    }

    info!("\n✅ Data pipeline demo complete!");

    // Clean shutdown
    kraken_handle.abort();
    polymarket_handle.abort();

    Ok(())
}
