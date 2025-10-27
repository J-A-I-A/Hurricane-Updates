import os
import torch
import glob
import time
import requests
import uuid
import shutil
import runpod
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from azure.storage.blob import BlobServiceClient, ContentSettings

# --- CONFIGURATION ---

# ODPEM page to scrape
BULLETIN_URL = "https://www.odpem.org.jm/weather-alert-melissa/"

# Azure Configuration (injected by RunPod)
AZURE_CONNECTION_STRING = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
AZURE_CONTAINER_NAME = os.environ.get('AZURE_CONTAINER_NAME')
STATE_FILE_BLOB_NAME = "processed_bulletins.txt" # The file in Azure to track processed URLs

# Model Configuration
MODEL_ID = "deepseek-ai/DeepSeek-OCR"
HF_TOKEN = os.environ.get('HF_TOKEN') # Optional: for gated models/private repos

# Global variables for the "hot" model
model = None
tok = None

# --- AZURE BLOB STORAGE FUNCTIONS ---

def get_azure_blob_client(blob_name):
    """Initializes and returns a BlobClient for a specific blob."""
    if not AZURE_CONNECTION_STRING:
        raise ValueError("ERROR: AZURE_STORAGE_CONNECTION_STRING not set.")
    if not AZURE_CONTAINER_NAME:
        raise ValueError("ERROR: AZURE_CONTAINER_NAME not set.")
        
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    return blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=blob_name)

def get_processed_urls() -> set:
    """Fetches the list of processed URLs from Azure Blob Storage."""
    print(f"Fetching state file: {STATE_FILE_BLOB_NAME}")
    try:
        client = get_azure_blob_client(STATE_FILE_BLOB_NAME)
        downloader = client.download_blob()
        data = downloader.readall().decode('utf-8')
        urls = set(line.strip() for line in data.splitlines() if line.strip())
        print(f"Found {len(urls)} processed URLs in state file.")
        return urls
    except Exception as e:
        # If blob doesn't exist, return an empty set
        print(f"State file not found or empty (this is normal on first run): {e}")
        return set()

def add_processed_url(url: str):
    """Appends a new URL to the processed list in Azure Blob Storage."""
    print(f"Adding URL to state file: {url}")
    try:
        client = get_azure_blob_client(STATE_FILE_BLOB_NAME)
        
        # Download existing content
        try:
            downloader = client.download_blob()
            existing_data = downloader.readall().decode('utf-8')
        except Exception:
            existing_data = "" # File doesn't exist yet

        # Append new URL
        new_data = existing_data + "\n" + url
        
        # Upload new content, overwriting the old file
        client.upload_blob(new_data, overwrite=True, content_settings=ContentSettings(content_type='text/plain'))
        print("State file updated successfully.")

    except Exception as e:
        print(f"CRITICAL: Failed to update state file in Azure: {e}")

# --- WEB SCRAPING FUNCTIONS ---

def find_all_bulletin_images(url: str) -> list[str]:
    """
    Scrapes the webpage to find ALL images in the main content.
    Returns a list of full image URLs.
    """
    # ... (This function is unchanged from your version) ...
    print(f"Scraping {url} for all images...")
    image_urls = []
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the main article content area
        content_area = soup.find_all('a', class_='elementor-button elementor-button-link elementor-size-sm')

        for link in content_area:
           if 'bulletin' in link['href'].lower():
            image_url = link['href']
            image_urls.append(image_url)
        return image_urls

            
    except requests.RequestException as e:
        print(f"Error fetching webpage: {e}")
        return []

def download_image_to_tmp(image_url: str) -> str | None:
    """Downloads an image to a temporary path and returns the path."""
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()
        
        # Get file extension
        file_extension = os.path.splitext(image_url)[1] or '.jpg'
        # Create a unique temp file path
        temp_file_path = f"/tmp/{uuid.uuid4()}{file_extension}"
        
        with open(temp_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"Downloaded image to {temp_file_path}")
        return temp_file_path
        
    except requests.RequestException as e:
        print(f"ERROR: Failed to download image {image_url}: {e}")
        return None

# --- OCR & AZURE UPLOAD FUNCTIONS ---

def upload_text_to_azure(text: str, blob_name: str):
    """Uploads the extracted text as a markdown file to Azure."""
    print(f"Uploading markdown to Azure as blob: {blob_name}")
    try:
        client = get_azure_blob_client(blob_name)
        # Set content type to markdown so it renders nicely in browsers
        md_settings = ContentSettings(content_type='text/markdown; charset=utf-8')
        
        client.upload_blob(text.encode('utf-8'), overwrite=True, content_settings=md_settings)
        print(f"Successfully uploaded {blob_name}")
        
    except Exception as e:
        print(f"ERROR: Failed to upload markdown to Azure: {e}")
        # Re-raise to be caught by the main handler
        raise

@torch.inference_mode()
def run_ocr(img_path: str) -> str:
    """
    Runs DeepSeek-OCR on a local image file and returns the extracted markdown.
    This function writes to a unique temp directory to avoid conflicts.
    """
    global model, tok
    print(f"Starting OCR for {img_path}...")
    t = time.time()

    # 1. Create a unique output directory for this specific run
    output_dir = f"/tmp/ocr-job-{uuid.uuid4()}"
    os.makedirs(output_dir, exist_ok=True)
    
    # The prompt to instruct the model to return markdown
    prompt = "<image>\n<|grounding|>Convert the document to markdown."
    
    try:
        # 2. Run inference, saving results to the unique directory
        # The returned 'res' object is NOT the text, as per user feedback.
        model.infer(
            tok,
            prompt=prompt,
            image_file=img_path,
            output_path=output_dir, # <-- Set to unique dir
            base_size=768,       
            image_size=512,      
            crop_mode=False,     
            save_results=True,   # <-- MUST be True to write file
            test_compress=False
        )
        
        print(f"OCR inference complete in {time.time()-t:.1f}s")
        
        # 3. Find the markdown file in the output directory
        # --- FIX ---
        # The model saves output based on the *input* filename.
        base_filename = os.path.basename(img_path).rsplit('.', 1)[0]
        markdown_file_path = os.path.join(output_dir, f"{base_filename}.md")
        
        if not os.path.exists(markdown_file_path):
            # Fallback: check for *any* .md file, just in case.
            md_files = glob.glob(os.path.join(output_dir, "*.md"))
            if not md_files:
                raise FileNotFoundError(f"OCR ran, but no .md file was found in {output_dir}")
            markdown_file_path = md_files[0] # Use the first one found
        
        print(f"Reading markdown from: {markdown_file_path}")
        # 4. Read the markdown text
        with open(markdown_file_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()

        if not markdown_text or markdown_text.isspace():
            raise ValueError("OCR ran but the output file was empty.")
            
        return markdown_text

    except Exception as e:
        print(f"Error during OCR inference: {e}")
        # Re-raise to be caught by the main handler
        raise
    finally:
        # 5. Clean up the unique directory and its contents
        if os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
                print(f"Cleaned up temp directory: {output_dir}")
            except Exception as e:
                print(f"Warning: Failed to clean up temp dir {output_dir}: {e}")

# --- MODEL INITIALIZATION & RUNPOD HANDLER ---

def initialize_model():
    """
    Loads the model and tokenizer into global variables.
    This runs once when the RunPod worker "hot" starts.
    """
    global model, tok
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Running on CPU (will be very slow).")
        device = "cpu"
        dtype = torch.float32
    else:
        print("CUDA is available. Loading model to GPU.")
        device = "cuda"
        dtype = torch.bfloat16

    print(f"Loading tokenizer: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_auth_token=HF_TOKEN)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    print("Tokenizer loaded. Loading model...")
    t0 = time.time()
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_safetensors=True,
        attn_implementation="eager" # Required for this arch on some platforms
    ).to(dtype=dtype, device=device).eval()
    
    print(f"Model loaded in {time.time()-t0:.1f}s. Worker is ready.")


def handler(event):
    """
    This is the main function called by RunPod for each serverless request.
    """
    global model, tok
    
    # Check if model is loaded (handles "cold" starts)
    if not model or not tok:
        print("Model not loaded (cold start). Initializing...")
        initialize_model()

    print("Handler invoked. Starting bulletin check...")
    
    try:
        # 1. Get the list of all images on the page
        all_image_urls = find_all_bulletin_images(BULLETIN_URL)
        if not all_image_urls:
            return {"status": "success", "message": "No images found on page."}

        # 2. Get the list of images we've already processed
        processed_urls = get_processed_urls()
        
        # 3. Determine which images are new
        new_images_to_process = [url for url in all_image_urls if url not in processed_urls]
        
        if not new_images_to_process:
            print("No new images found. Job complete.")
            return {"status": "success", "message": "No new images to process."}

        print(f"Found {len(new_images_to_process)} new image(s) to process.")
        
        processed_count = 0
        failed_count = 0
        
        # 4. Process each new image
        for image_url in new_images_to_process:
            local_image_path = None
            try:
                # 4a. Download the image
                local_image_path = download_image_to_tmp(image_url)
                if not local_image_path:
                    raise Exception(f"Failed to download image: {image_url}")

                # 4b. Run OCR to get markdown
                markdown_text = run_ocr(local_image_path)
                
                # 4c. Upload markdown to Azure
                # Create a blob name from the image URL, e.g., "Bulletin-No-7.md"
                blob_name = os.path.basename(image_url).split('?')[0] + ".md"
                upload_text_to_azure(markdown_text, blob_name)
                
                # 4d. If upload succeeds, add to our processed list
                add_processed_url(image_url)
                processed_count += 1
                
            except Exception as e:
                print(f"Failed to process image {image_url}: {e}")
                failed_count += 1
            finally:
                # 4e. Clean up the local downloaded image
                if local_image_path and os.path.exists(local_image_path):
                    os.remove(local_image_path)
                    
        message = f"Job complete. Processed {processed_count} new images. Failed {failed_count}."
        print(message)
        return {"status": "success", "message": message}

    except Exception as e:
        print(f"Unhandled error in handler: {e}")
        return {"status": "error", "message": str(e)}

# --- START THE SERVERLESS WORKER ---
if __name__ == "__main__":
    print("Starting RunPod serverless worker...")
    # Initialize the model *before* starting the server
    # This makes the worker "hot"
    initialize_model()
    runpod.serverless.start({"handler": handler})