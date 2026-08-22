use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use rand::Rng;
use tokio::sync::Semaphore;

/// Concurrency limiter + retry + circuit breaker for DashScope API calls.
pub struct ConcurrencyControl {
    semaphore: Arc<Semaphore>,
    timeout: Duration,
    max_retries: u32,
    base_delay: Duration,
    max_delay: Duration,
    /// Consecutive failure counter for circuit breaker.
    consecutive_failures: AtomicU32,
    /// Timestamp (ms) when circuit breaker tripped, 0 = closed.
    tripped_at_ms: AtomicU32,
    failure_threshold: u32,
    cooldown: Duration,
}

impl ConcurrencyControl {
    pub fn new(max_concurrency: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrency)),
            timeout: Duration::from_secs(30),
            max_retries: 3,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
            consecutive_failures: AtomicU32::new(0),
            tripped_at_ms: AtomicU32::new(0),
            failure_threshold: 5,
            cooldown: Duration::from_secs(60),
        }
    }

    /// Acquire a concurrency permit. Drop the guard to release.
    pub async fn acquire(&self) -> Result<tokio::sync::OwnedSemaphorePermit, String> {
        self.semaphore.clone()
            .acquire_owned()
            .await
            .map_err(|_| "Semaphore closed".to_string())
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Check if the circuit breaker is tripped (fail-fast mode).
    pub fn is_tripped(&self) -> bool {
        let ts = self.tripped_at_ms.load(Ordering::Acquire);
        if ts == 0 {
            return false;
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u32;
        if now.saturating_sub(ts) < self.cooldown.as_millis() as u32 {
            return true;
        }
        // Cooldown expired, reset
        self.tripped_at_ms.store(0, Ordering::Release);
        false
    }

    /// Record a successful API call — reset failure counter.
    pub fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::Release);
    }

    /// Record a failed API call — increment counter, maybe trip breaker.
    pub fn record_failure(&self) {
        let count = self.consecutive_failures.fetch_add(1, Ordering::AcqRel) + 1;
        if count >= self.failure_threshold {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u32;
            self.tripped_at_ms.store(now, Ordering::Release);
            tracing::warn!("Circuit breaker tripped after {} consecutive failures", count);
        }
    }

    /// Calculate jittered exponential backoff delay for a given attempt (0-indexed).
    pub fn backoff_delay(&self, attempt: u32) -> Duration {
        let base_ms = self.base_delay.as_millis() as u64;
        let exp = base_ms.checked_shl(attempt).unwrap_or(self.max_delay.as_millis() as u64);
        let capped = exp.min(self.max_delay.as_millis() as u64);
        let jitter = rand::thread_rng().gen_range(0..=base_ms);
        Duration::from_millis(capped + jitter)
    }

    pub fn max_retries(&self) -> u32 {
        self.max_retries
    }
}
