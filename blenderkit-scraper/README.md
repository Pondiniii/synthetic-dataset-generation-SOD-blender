# BlenderKit Scraper

Fast Rust CLI for scraping and downloading free assets from [BlenderKit](https://www.blenderkit.com/) API.

## Features

- **Free Assets Only**: Filters for `isFree` or `canDownload` assets
- **Smart HDR Downloads**: Gets 2K .exr files (not JPEG thumbnails)
- **Progress Bars**: Real-time download tracking with size/speed
- **Size Limits**: Control total download size and asset count
- **.env Support**: Secure API key storage

## Installation

```bash
cargo build --release
```

Binary will be at: `target/release/blenderkit-scraper`

## Setup

### 1. Get BlenderKit API Key

1. Create account at [blenderkit.com](https://www.blenderkit.com/)
2. Get API key from [API Keys page](https://www.blenderkit.com/api_keys/)

### 2. Configure

```bash
echo 'BLENDERKIT_API_KEY=your_key_here' > .env
```

## Usage

### 1. Scrape Assets

```bash
# Scrape 50 free models
cargo run --release -- scrape --asset-type model --limit 50 --output models.json

# Scrape 20 HDRs
cargo run --release -- scrape --asset-type hdr --limit 20 --output hdrs.json
```

### 2. View Info

```bash
cargo run --release -- info --input models.json
```

### 3. Download Assets

```bash
# Download all (with confirmation)
cargo run --release -- download --input models.json --output downloads

# Download with limits (skip confirmation)
cargo run --release -- download \
    --input models.json \
    --output downloads \
    --max-size 1000 \
    --limit 10 \
    --yes
```

**Output**: `downloads/model/*.blend` and `downloads/hdr/*.exr`

## Examples

### Quick Test (1 model + 1 HDR)

```bash
cargo run -r -- scrape --asset-type model --limit 1 --output model.json
cargo run -r -- scrape --asset-type hdr --limit 1 --output hdr.json
cargo run -r -- download --input model.json --output downloads --yes
cargo run -r -- download --input hdr.json --output downloads --yes
```

### Production (100 models + 20 HDRs with size limit)

```bash
# Scrape
cargo run -r -- scrape --asset-type model --limit 100 --output models.json
cargo run -r -- scrape --asset-type hdr --limit 20 --output hdrs.json

# Check size
cargo run -r -- info --input models.json

# Download with 5GB limit
cargo run -r -- download --input models.json --output downloads --max-size 5000 --yes
cargo run -r -- download --input hdrs.json --output downloads --yes
```

## Commands

### `scrape`
- `-a, --asset-type <TYPE>` - Asset type: model, hdr, material (default: model)
- `-o, --output <FILE>` - Output JSON file (default: scraped_assets.json)
- `-l, --limit <N>` - Limit number of assets

### `info`
- `-i, --input <FILE>` - Input JSON file (default: scraped_assets.json)

### `download`
- `-i, --input <FILE>` - Input JSON from scrape
- `-o, --output <DIR>` - Output directory (default: downloads)
- `-m, --max-size <MB>` - Maximum total size in MB
- `-l, --limit <N>` - Maximum number of assets
- `-y, --yes` - Skip confirmation

## Output Structure

```
downloads/
├── model/
│   ├── {uuid}.blend        # 3D models
│   └── ...
└── hdr/
    ├── {uuid}.exr          # 2K HDR environments (not JPEG!)
    └── ...
```

**Note**: HDRs download as `resolution_2K` (.exr, 32-bit float), NOT JPEG thumbnails.
