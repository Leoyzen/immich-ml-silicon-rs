use std::net::SocketAddr;
use std::sync::Arc;

use axum::{Router, routing::{get, post}, extract::DefaultBodyLimit};

mod config;
mod concurrency;
mod schema;
mod state;
mod routes;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cfg = config::Config::from_env()?;
    tracing::info!("Config: port={}, cache_dir={}, device={}, max_concurrency={}",
        cfg.port, cfg.cache_dir.display(), cfg.device, cfg.max_concurrency);

    let state = Arc::new(state::AppState::new(cfg.clone()).await?);

    let app = Router::new()
        .route("/ping", get(routes::ping))
        .route("/predict", post(routes::predict))
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024)) // 64MB, matching immich-ml's spool_max_size
        .with_state(state);

    let addr: SocketAddr = format!("0.0.0.0:{}", cfg.port).parse()?;
    tracing::info!("immich-ml-rust listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;

    // Graceful shutdown: finish in-flight requests on SIGTERM/SIGINT
    let shutdown = async {
        let ctrl_c = async {
            tokio::signal::ctrl_c().await.ok();
        };
        #[cfg(unix)]
        let terminate = async {
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .unwrap()
                .recv()
                .await;
        };
        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();
        tokio::select! {
            _ = ctrl_c => {},
            _ = terminate => {},
        }
        tracing::info!("Shutdown signal received, draining in-flight requests...");
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;

    tracing::info!("Server stopped.");
    Ok(())
}
