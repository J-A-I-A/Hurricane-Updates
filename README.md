# Hurricane Updates - Bulletin OCR Service

An automated service that:
- Monitors the Office of Disaster Preparedness and Emergency Management (ODPEM) Jamaica weather alerts website, extracts bulletin images, performs OCR using DeepSeek-OCR, and stores the results as markdown in Azure Blob Storage.
- Fetches the latest news stories from Jamaica Information Service (JIS) and uploads them as markdown to a separate Azure container.

## Overview

This service periodically scrapes the ODPEM weather alert webpage for bulletin images, performs OCR on them to extract text content, and uploads the extracted markdown to Azure Blob Storage. It also fetches recent JIS news articles, converts the content to markdown, and uploads them to a separate Azure container. The service is designed to run on RunPod's serverless GPU platform for efficient processing.

## Features

- **Automated Web Scraping (Bulletins)**: Automatically finds and extracts all bulletin images from the ODPEM website
- **Automated News Fetching**: Scrapes latest stories from JIS and converts them to markdown
- **State-of-the-Art OCR**: Uses DeepSeek-OCR model for accurate text extraction from images
- **Duplicate Prevention**: Tracks processed images to avoid reprocessing
- **Markdown Output**: Converts images to well-structured markdown format
- **Azure Blob Storage Integration**: Uploads bulletins and news to separate containers
- **Standardized Bulletin Naming**: Bulletins saved as `Bulletin-<number>.md` where number is 1 + the highest number detected across existing `.md` filenames (trailing digits)
- **Scheduled Execution**: Runs hourly via GitHub Actions
- **Serverless Architecture**: Built for RunPod's serverless GPU platform

## Architecture

```
GitHub Actions (hourly trigger)
    ↓
RunPod Serverless Endpoint
    ↓
Scrape ODPEM images → OCR → Upload to `AZURE_CONTAINER_NAME`
    ↘
     Fetch JIS news → HTML→MD → Upload to `AZURE_NEWS_CONTAINER_NAME`
```

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (required for DeepSeek-OCR)
- Azure Blob Storage account
- RunPod account

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Set the following environment variables (or add them as secrets in RunPod):

- `AZURE_STORAGE_CONNECTION_STRING`: Your Azure Storage connection string
- `AZURE_CONTAINER_NAME`: Container for bulletin markdown files
- `AZURE_NEWS_CONTAINER_NAME`: Container for news markdown files
- `HF_TOKEN` (optional): Hugging Face token if needed

### 3. RunPod Deployment

Build and deploy the Docker container to RunPod:

```bash
docker build -t bulletin-ocr-service .
```

Configure in RunPod:
1. Create a serverless endpoint
2. Add the Docker image
3. Set environment variables/secrets
4. Configure GPU requirements (recommended: 1x A40 or equivalent)

### 4. GitHub Actions Setup

Add the following secrets to your GitHub repository:

- `RUNPOD_ENDPOINT_ID`: Your RunPod endpoint ID
- `RUNPOD_API_KEY`: Your RunPod API key

The workflow will automatically trigger hourly, or can be run manually from the Actions tab.

## Usage

### Manual Trigger

The service can be triggered manually by sending a POST request to your RunPod endpoint:

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "scrape_url": "https://www.odpem.org.jm/weather-alert-melissa/"
    }
  }'
```

### Custom Configuration

You can override the default configuration via the handler input:

```json
{
  "input": {
    "container_name": "your-container-name",
    "scrape_url": "https://custom-url.com/alerts"
  }
}
```

## How It Works

1. **Bulletins**
   - Scrape ODPEM and collect image links
   - Compare to a state file in Azure to skip processed images
   - Download new images and run DeepSeek-OCR to produce markdown
   - Compute next bulletin number by scanning all `.md` filenames and extracting the last number; upload as `Bulletin-<number>.md`
   - Append processed image URLs to the state file in Azure
2. **News**
   - Scrape latest articles from `https://jis.gov.jm`
   - Parse HTML content and convert to markdown
   - Upload each article markdown to `AZURE_NEWS_CONTAINER_NAME`
3. The handler always runs both flows; even if there are no new bulletin images, news fetching still runs.

## Output

Bulletins are uploaded as files named:

```
Bulletin-<number>.md
```

The `<number>` is 1 greater than the highest number found in any existing `.md` filename (by scanning trailing/last digits in filenames). News files are uploaded as `<date>_<slugified-title>.md` to the news container.

## Configuration

Key configuration variables in `bulletin_ocr_service.py`:

- `BULLETIN_URL`: ODPEM page to scrape for bulletin images
- `AZURE_CONTAINER_NAME`: Azure container for bulletins
- `AZURE_NEWS_CONTAINER_NAME`: Azure container for news
- `STATE_FILE_BLOB_NAME`: Blob filename used to track processed bulletin image URLs

## Technologies

- **Python**: Core implementation
- **DeepSeek-OCR**: OCR model for text extraction
- **PyTorch**: Deep learning framework
- **RunPod**: Serverless GPU platform
- **Azure Blob Storage**: Cloud storage for outputs
- **BeautifulSoup**: Web scraping
- **markdownify**: HTML-to-Markdown conversion for news
- **GitHub Actions**: CI/CD and scheduling

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
