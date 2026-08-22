use std::sync::Arc;
use std::time::Duration;

use immich_ml_backends::{BackendError, ClipBackend};
use tokio::sync::{mpsc, oneshot};

struct BatchRequest {
    image_bytes: Vec<u8>,
    response_tx: oneshot::Sender<Result<Vec<f32>, BackendError>>,
}

pub struct ClipBatcher {
    tx: mpsc::Sender<BatchRequest>,
}

impl ClipBatcher {
    pub fn new(
        clip_client: Arc<dyn ClipBackend>,
        max_batch: usize,
        flush_interval_ms: u64,
    ) -> Self {
        let (tx, mut rx) = mpsc::channel::<BatchRequest>(200);

        tokio::spawn(async move {
            let mut buffer: Vec<BatchRequest> = Vec::with_capacity(max_batch);
            let mut interval = tokio::time::interval(Duration::from_millis(flush_interval_ms));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                tokio::select! {
                    Some(req) = rx.recv() => {
                        buffer.push(req);
                        // Drain all immediately-available items (burst absorption)
                        while buffer.len() < max_batch {
                            match rx.try_recv() {
                                Ok(r) => buffer.push(r),
                                Err(_) => break,
                            }
                        }
                        if buffer.len() >= max_batch {
                            flush(&mut buffer, &clip_client).await;
                            interval.reset();
                        }
                    }
                    _ = interval.tick() => {
                        if !buffer.is_empty() {
                            flush(&mut buffer, &clip_client).await;
                        }
                    }
                }
            }
        });

        Self { tx }
    }

    pub async fn submit(&self, image_bytes: Vec<u8>) -> Result<Vec<f32>, BackendError> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.tx
            .send(BatchRequest {
                image_bytes,
                response_tx: resp_tx,
            })
            .await
            .map_err(|_| BackendError::Other("CLIP batcher channel closed".into()))?;
        resp_rx
            .await
            .map_err(|_| BackendError::Other("CLIP batcher dropped response".into()))?
    }
}

async fn flush(buffer: &mut Vec<BatchRequest>, clip_client: &Arc<dyn ClipBackend>) {
    if buffer.is_empty() {
        return;
    }

    let count = buffer.len();
    tracing::info!("CLIP batch flush: {} images", count);

    // Extract image bytes
    let images: Vec<Vec<u8>> = buffer.iter().map(|r| r.image_bytes.clone()).collect();

    match clip_client.encode_image_batch(&images).await {
        Ok(embeddings) => {
            if embeddings.len() != count {
                tracing::error!(
                    "CLIP batch mismatch: sent {} images, got {} embeddings",
                    count,
                    embeddings.len()
                );
                for req in buffer.drain(..) {
                    let _ = req
                        .response_tx
                        .send(Err(BackendError::Other("Batch result count mismatch".into())));
                }
                return;
            }
            for (req, emb) in buffer.drain(..).zip(embeddings) {
                let _ = req.response_tx.send(Ok(emb));
            }
        }
        Err(e) => {
            tracing::error!("CLIP batch failed ({} images): {}", count, e);
            let err_msg = e.to_string();
            for req in buffer.drain(..) {
                let _ = req
                    .response_tx
                    .send(Err(BackendError::Other(err_msg.clone())));
            }
        }
    }
}
