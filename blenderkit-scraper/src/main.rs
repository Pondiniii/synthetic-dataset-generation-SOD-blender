use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use dotenv::dotenv;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

#[derive(Parser)]
#[command(name = "blenderkit-scraper")]
#[command(about = "BlenderKit free assets scraper", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Phase 1: Scrape metadata from BlenderKit API
    Scrape {
        /// Asset type: model, hdr, material
        #[arg(short, long, default_value = "model")]
        asset_type: String,

        /// Output JSON file
        #[arg(short, long, default_value = "scraped_assets.json")]
        output: PathBuf,

        /// Maximum number of assets to scrape (None = all)
        #[arg(short, long)]
        limit: Option<usize>,
    },

    /// Show info about scraped assets
    Info {
        /// Input JSON file from scrape command
        #[arg(short, long, default_value = "scraped_assets.json")]
        input: PathBuf,
    },

    /// Phase 2: Download assets
    Download {
        /// Input JSON file from scrape command
        #[arg(short, long, default_value = "scraped_assets.json")]
        input: PathBuf,

        /// Output directory for downloads
        #[arg(short, long, default_value = "downloads")]
        output: PathBuf,

        /// Maximum total size in MB (None = no limit)
        #[arg(short, long)]
        max_size: Option<u64>,

        /// Maximum number of assets to download
        #[arg(short, long)]
        limit: Option<usize>,

        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct Asset {
    asset_base_id: String,
    name: String,
    asset_type: String,
    files_size: Option<u64>,
    files: Vec<FileInfo>,
    #[serde(default)]
    is_free: bool,
    #[serde(default)]
    can_download: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct FileInfo {
    file_type: String,
    download_url: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SearchResponse {
    count: u64,
    next: Option<String>,
    results: Vec<Asset>,
}

#[derive(Debug, Deserialize)]
struct DownloadResponse {
    #[serde(rename = "filePath")]
    file_path: String,
}

struct BlenderKitClient {
    client: Client,
    api_key: String,
    base_url: String,
}

impl BlenderKitClient {
    fn new(api_key: String) -> Self {
        let client = Client::builder()
            .user_agent("BlenderKit-Scraper-Rust/1.0")
            .build()
            .unwrap();

        Self {
            client,
            api_key,
            base_url: "https://www.blenderkit.com/api/v1".to_string(),
        }
    }

    async fn search_all(&self, asset_type: &str, limit: Option<usize>) -> Result<Vec<Asset>> {
        let mut all_assets = Vec::new();
        let mut page = 1;
        let mut next_url: Option<String> = None;

        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} [{elapsed}] {msg}")
                .unwrap(),
        );

        loop {
            pb.set_message(format!("Page {} - {} assets found", page, all_assets.len()));

            let url = if let Some(ref next) = next_url {
                if next.starts_with("http://") || next.starts_with("https://") {
                    next.clone()
                } else {
                    format!("{}{}", self.base_url, next)
                }
            } else {
                format!(
                    "{}/search/?query=+asset_type:{}+order:-is_free,-last_blend_upload&dict_parameters=1&page_size=100",
                    self.base_url, asset_type
                )
            };

            let response = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .await?;

            let status = response.status();
            let body = response.text().await?;

            if !status.is_success() {
                bail!(
                    "BlenderKit responded with status {} for {} (snippet: {})",
                    status,
                    url,
                    body_snippet(&body)
                );
            }

            let response: SearchResponse = serde_json::from_str(&body).context(
                format!(
                    "Failed to decode BlenderKit response (url: {}, body snippet: {})",
                    url,
                    body_snippet(&body)
                )
            )?;

            if page == 1 {
                pb.println(format!("📊 Total available: {} assets", response.count));
            }

            let free_count = response
                .results
                .iter()
                .filter(|a| a.is_free || a.can_download)
                .count();

            // Stop if no free assets in page
            if free_count == 0 && !all_assets.is_empty() {
                pb.println("✋ No more free assets");
                break;
            }

            // Add free assets
            for asset in response.results {
                if asset.is_free || asset.can_download {
                    all_assets.push(asset);

                    if let Some(lim) = limit {
                        if all_assets.len() >= lim {
                            pb.finish_with_message(format!("✅ Reached limit: {} assets", lim));
                            return Ok(all_assets);
                        }
                    }
                }
            }

            pb.println(format!("📄 Page {}: +{} free", page, free_count));

            // Check for next page
            if response.next.is_none() {
                pb.finish_with_message("✅ Reached last page");
                break;
            }

            next_url = response.next;
            page += 1;
        }

        pb.finish_with_message(format!("✨ Found {} free assets", all_assets.len()));
        Ok(all_assets)
    }

    async fn download_asset(
        &self,
        download_url: &str,
        output_path: &PathBuf,
        pb: &ProgressBar,
    ) -> Result<()> {
        // Step 1: Get presigned URL
        let scene_uuid = Uuid::new_v4();
        let url_with_uuid = if download_url.starts_with('/') {
            format!(
                "{}{}?scene_uuid={}",
                self.base_url, download_url, scene_uuid
            )
        } else {
            format!("{}?scene_uuid={}", download_url, scene_uuid)
        };

        let download_resp = self
            .client
            .get(&url_with_uuid)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?
            .json::<DownloadResponse>()
            .await?;

        // Step 2: Download from presigned URL
        let response = self.client.get(&download_resp.file_path).send().await?;
        let total_size = response.content_length().unwrap_or(0);

        pb.set_length(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        // Create parent dir
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut file = tokio::fs::File::create(output_path).await?;
        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();

        use futures::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;
            pb.set_position(downloaded);
        }

        file.flush().await?;
        Ok(())
    }
}

fn format_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_idx])
}

fn body_snippet(body: &str) -> String {
    let mut snippet: String = body.chars().take(200).collect();
    if body.chars().count() > 200 {
        snippet.push('…');
    }
    snippet.replace('\n', " ")
}

async fn cmd_scrape(asset_type: String, output: PathBuf, limit: Option<usize>) -> Result<()> {
    println!("🔍 Scraping {} assets from BlenderKit...", asset_type);

    let api_key = env::var("BLENDERKIT_API_KEY")
        .context("BLENDERKIT_API_KEY not found in environment")?;

    let client = BlenderKitClient::new(api_key);
    let assets = client.search_all(&asset_type, limit).await?;

    // Save to JSON
    let json = serde_json::to_string_pretty(&assets)?;
    fs::write(&output, json)?;

    println!("\n💾 Saved {} assets to: {}", assets.len(), output.display());
    println!("📊 Total size: {}", format_size(
        assets.iter().filter_map(|a| a.files_size).sum()
    ));

    Ok(())
}

async fn cmd_info(input: PathBuf) -> Result<()> {
    let content = fs::read_to_string(&input)
        .context(format!("Failed to read {}", input.display()))?;
    let assets: Vec<Asset> = serde_json::from_str(&content)?;

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║              SCRAPED ASSETS INFO                         ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Total assets:    {:>40} ║", assets.len());

    let total_size: u64 = assets.iter().filter_map(|a| a.files_size).sum();
    println!("║ Total size:      {:>40} ║", format_size(total_size));

    let avg_size = if !assets.is_empty() {
        total_size / assets.len() as u64
    } else {
        0
    };
    println!("║ Average size:    {:>40} ║", format_size(avg_size));

    // Count by type
    let mut type_counts = std::collections::HashMap::new();
    for asset in &assets {
        *type_counts.entry(&asset.asset_type).or_insert(0) += 1;
    }

    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Asset Types:                                             ║");
    for (typ, count) in type_counts {
        println!("║   {:<20} {:>30} ║", typ, count);
    }

    println!("╚══════════════════════════════════════════════════════════╝");

    Ok(())
}

async fn cmd_download(
    input: PathBuf,
    output: PathBuf,
    max_size: Option<u64>,
    limit: Option<usize>,
    yes: bool,
) -> Result<()> {
    let content = fs::read_to_string(&input)?;
    let mut assets: Vec<Asset> = serde_json::from_str(&content)?;

    // Apply limit
    if let Some(lim) = limit {
        assets.truncate(lim);
    }

    // Calculate total size
    let mut total_size: u64 = assets.iter().filter_map(|a| a.files_size).sum();

    // Apply max_size filter
    if let Some(max_mb) = max_size {
        let max_bytes = max_mb * 1024 * 1024;
        let mut cumulative = 0u64;
        assets.retain(|a| {
            if let Some(size) = a.files_size {
                if cumulative + size <= max_bytes {
                    cumulative += size;
                    return true;
                }
            }
            false
        });
        total_size = cumulative;
    }

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║              DOWNLOAD PLAN                               ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Assets to download: {:>37} ║", assets.len());
    println!("║ Total size:         {:>37} ║", format_size(total_size));
    println!("║ Output directory:   {:>37} ║", output.display());
    println!("╚══════════════════════════════════════════════════════════╝");

    if !yes {
        print!("\n⚠️  Proceed with download? [y/N]: ");
        use std::io::{self, Write};
        io::stdout().flush()?;

        let mut response = String::new();
        io::stdin().read_line(&mut response)?;

        if !response.trim().eq_ignore_ascii_case("y") {
            println!("❌ Cancelled");
            return Ok(());
        }
    }

    println!("\n📥 Starting download...\n");

    let api_key = env::var("BLENDERKIT_API_KEY")?;
    let client = BlenderKitClient::new(api_key);

    let mut successful = 0;
    let mut failed = 0;

    for (i, asset) in assets.iter().enumerate() {
        let pb = ProgressBar::new(0);
        pb.set_message(format!("[{}/{}] {}", i + 1, assets.len(), asset.name));

        // Find download URL based on asset type
        let (download_url, file_ext) = if asset.asset_type == "hdr" {
            // For HDR: prefer resolution_2K (real HDRI), fallback to blend
            let url = asset
                .files
                .iter()
                .find(|f| f.file_type == "resolution_2K")
                .or_else(|| asset.files.iter().find(|f| f.file_type == "blend"))
                .and_then(|f| f.download_url.as_ref());
            (url, "exr")
        } else {
            // For models: use blend file
            let url = asset
                .files
                .iter()
                .find(|f| f.file_type == "blend")
                .and_then(|f| f.download_url.as_ref());
            (url, "blend")
        };

        if let Some(url) = download_url {
            let file_path = output
                .join(&asset.asset_type)
                .join(format!("{}.{}", asset.asset_base_id, file_ext));

            match client.download_asset(url, &file_path, &pb).await {
                Ok(_) => {
                    pb.finish_with_message(format!("✅ {}", asset.name));
                    successful += 1;
                }
                Err(e) => {
                    pb.finish_with_message(format!("❌ {} - Error: {}", asset.name, e));
                    failed += 1;
                }
            }
        } else {
            pb.finish_with_message(format!("⚠️  {} - No download URL", asset.name));
            failed += 1;
        }
    }

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║              DOWNLOAD SUMMARY                            ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Successful: {:>45} ║", successful);
    println!("║ Failed:     {:>45} ║", failed);
    println!("╚══════════════════════════════════════════════════════════╝");

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    let cli = Cli::parse();

    match cli.command {
        Commands::Scrape {
            asset_type,
            output,
            limit,
        } => cmd_scrape(asset_type, output, limit).await?,
        Commands::Info { input } => cmd_info(input).await?,
        Commands::Download {
            input,
            output,
            max_size,
            limit,
            yes,
        } => cmd_download(input, output, max_size, limit, yes).await?,
    }

    Ok(())
}
