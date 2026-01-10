// Circular buffer for managing time-series sequences

use chrono::{DateTime, Utc};
use ringbuffer::{AllocRingBuffer, RingBuffer};
use std::collections::HashMap;

use super::types::{Asset, AssetFeatures};

/// Circular buffer for storing time-series data
pub struct FeatureBuffer {
    /// Buffer for each asset
    buffers: HashMap<Asset, AllocRingBuffer<AssetFeatures>>,
    /// Maximum sequence length
    max_len: usize,
}

impl FeatureBuffer {
    /// Create a new feature buffer
    pub fn new(assets: Vec<Asset>, max_len: usize) -> Self {
        let mut buffers = HashMap::new();

        for asset in assets {
            buffers.insert(asset, AllocRingBuffer::new(max_len));
        }

        Self { buffers, max_len }
    }

    /// Push new features for an asset
    pub fn push(&mut self, features: AssetFeatures) {
        if let Some(buffer) = self.buffers.get_mut(&features.asset) {
            buffer.push(features);
        }
    }

    /// Get the current sequence for an asset
    pub fn get_sequence(&self, asset: Asset) -> Option<Vec<AssetFeatures>> {
        self.buffers.get(&asset).map(|buffer| {
            buffer.iter().cloned().collect()
        })
    }

    /// Get sequences for all assets as a batch
    /// Returns None if any asset doesn't have enough data
    pub fn get_batch(&self, min_len: usize) -> Option<HashMap<Asset, Vec<AssetFeatures>>> {
        let mut batch = HashMap::new();

        for (asset, buffer) in &self.buffers {
            if buffer.len() < min_len {
                return None; // Not enough data yet
            }

            batch.insert(*asset, buffer.iter().cloned().collect());
        }

        Some(batch)
    }

    /// Check if all assets have at least min_len samples
    pub fn is_ready(&self, min_len: usize) -> bool {
        self.buffers.values().all(|buffer| buffer.len() >= min_len)
    }

    /// Get buffer length for an asset
    pub fn len(&self, asset: Asset) -> usize {
        self.buffers.get(&asset).map_or(0, |b| b.len())
    }

    /// Check if buffer is empty for an asset
    pub fn is_empty(&self, asset: Asset) -> bool {
        self.len(asset) == 0
    }

    /// Get the maximum buffer capacity
    pub fn capacity(&self) -> usize {
        self.max_len
    }

    /// Clear all buffers
    pub fn clear(&mut self) {
        for buffer in self.buffers.values_mut() {
            buffer.clear();
        }
    }

    /// Get latest feature for an asset
    pub fn get_latest(&self, asset: Asset) -> Option<&AssetFeatures> {
        self.buffers.get(&asset)?.back()
    }

    /// Get oldest feature for an asset
    pub fn get_oldest(&self, asset: Asset) -> Option<&AssetFeatures> {
        self.buffers.get(&asset)?.front()
    }

    /// Get time range of buffered data for an asset
    pub fn time_range(&self, asset: Asset) -> Option<(DateTime<Utc>, DateTime<Utc>)> {
        let buffer = self.buffers.get(&asset)?;

        if buffer.is_empty() {
            return None;
        }

        let oldest = buffer.front()?.timestamp;
        let latest = buffer.back()?.timestamp;

        Some((oldest, latest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_buffer() {
        let assets = vec![Asset::BTC, Asset::ETH];
        let mut buffer = FeatureBuffer::new(assets, 10);

        // Initially empty
        assert!(!buffer.is_ready(5));
        assert_eq!(buffer.len(Asset::BTC), 0);

        // Add some features
        for i in 0..5 {
            let features = AssetFeatures::empty(Asset::BTC, Utc::now());
            buffer.push(features);
        }

        assert_eq!(buffer.len(Asset::BTC), 5);
        assert!(!buffer.is_ready(5)); // ETH still empty

        // Add to ETH
        for i in 0..5 {
            let features = AssetFeatures::empty(Asset::ETH, Utc::now());
            buffer.push(features);
        }

        assert!(buffer.is_ready(5));

        // Get batch
        let batch = buffer.get_batch(5).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[&Asset::BTC].len(), 5);
        assert_eq!(batch[&Asset::ETH].len(), 5);

        // Test overflow (buffer size is 10)
        for i in 0..10 {
            let features = AssetFeatures::empty(Asset::BTC, Utc::now());
            buffer.push(features);
        }

        // Should have exactly 10 elements
        assert_eq!(buffer.len(Asset::BTC), 10);
    }

    #[test]
    fn test_time_range() {
        let assets = vec![Asset::BTC];
        let mut buffer = FeatureBuffer::new(assets, 10);

        assert!(buffer.time_range(Asset::BTC).is_none());

        let t1 = Utc::now();
        buffer.push(AssetFeatures::empty(Asset::BTC, t1));

        std::thread::sleep(std::time::Duration::from_millis(10));

        let t2 = Utc::now();
        buffer.push(AssetFeatures::empty(Asset::BTC, t2));

        let (start, end) = buffer.time_range(Asset::BTC).unwrap();
        assert_eq!(start, t1);
        assert_eq!(end, t2);
    }
}
