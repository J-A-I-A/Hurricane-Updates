# Hurricane Updates - Bulletin OCR Service

An automated OCR service that monitors the Office of Disaster Preparedness and Emergency Management (ODPEM) Jamaica weather alerts website, extracts bulletin images, performs OCR using DeepSeek-OCR, and stores the results as markdown files in Azure Blob Storage.

## Overview

This service periodically scrapes the ODPEM weather alert webpage for bulletin images, performs OCR on them to extract text content, and automatically uploads the extracted markdown to Azure Blob Storage. The service is designed to run on RunPod's serverless GPU platform for efficient processing.

## Features

- **Automated Web Scraping**: Automatically finds and extracts all bulletin images from the ODPEM website
- **State-of-the-Art OCR**: Uses DeepSeek-OCR model for accurate text extraction from images
- **Duplicate Prevention**: Tracks processed images to avoid reprocessing
- **Markdown Output**: Converts images to well-structured markdown format
- **Azure Blob Storage Integration**: Automatically uploads results to Azure Blob Storage
- **Scheduled Execution**: Runs hourly via GitHub Actions
- **Serverless Architecture**: Built for RunPod's serverless GPU platform

## Architecture

```
GitHub Actions (hourly trigger)
    ↓
RunPod Serverless Endpoint
    ↓
Extract images from webpage
    ↓
DeepSeek-OCR Model (GPU-accelerated)
    ↓
Azure Blob Storage (markdown files)
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

- `AZURE_CONTAINER_NAME`: Name of your Azure Blob Storage container
- `AZURE_STORAGE_CONNECTION_STRING`: Your Azure Storage connection string

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

1. **Web Scraping**: The service visits the configured webpage and extracts all images from the main content area
2. **State Check**: Compares found images against a state file in Azure Blob Storage to identify new images
3. **Image Processing**: Downloads each new image and resizes if necessary for optimal OCR performance
4. **OCR Extraction**: Uses the DeepSeek-OCR model to extract text and convert it to markdown format
5. **Upload Results**: Saves the markdown output to Azure Blob Storage with a timestamp
6. **State Update**: Updates the processed URLs state file to prevent future duplicate processing

## Output

Markdown files are stored in Azure Blob Storage under the `bulletins/` prefix with the following naming pattern:

```
bulletins/YYYY-MM-DD_HH-MM-SS_image_name.md
```

## Configuration

Key configuration variables in `bulletin_ocr_service.py`:

- `WEBPAGE_URL`: Default webpage to scrape (can be overridden)
- `WORKING_DIR`: Local temporary directory for processing
- `STATE_FILE_NAME`: Name of the state file in blob storage

## Technologies

- **Python**: Core implementation
- **DeepSeek-OCR**: OCR model for text extraction
- **PyTorch**: Deep learning framework
- **RunPod**: Serverless GPU platform
- **Azure Blob Storage**: Cloud storage for outputs
- **BeautifulSoup**: Web scraping
- **GitHub Actions**: CI/CD and scheduling

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
