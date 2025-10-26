import os
import time
import glob
import torch
import requests
from PIL import Image
from bs4 import BeautifulSoup
from transformers import AutoModel, AutoTokenizer
from urllib.parse import urljoin

# --- Azure Blob Storage Imports ---
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import AzureError

# --- CONFIGURATION ---
# TODO: Set your Azure container name here
AZURE_CONTAINER_NAME = "updates"
# TODO: Set your AZURE_STORAGE_CONNECTION_STRING as an environment variable
AZURE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

# The webpage to scrape
WEBPAGE_URL = "https://www.odpem.org.jm/weather-alert-melissa/"

# How often to check for a new image (in seconds)
POLL_INTERVAL_SECONDS = 3600  # 1 hour

# Local storage setup
WORKING_DIR = "/tmp/bulletin_ocr"
STATE_FILE = os.path.join(WORKING_DIR, "last_processed_url.txt")
LOCAL_IMG_PATH = os.path.join(WORKING_DIR, "bulletin_image.jpg")
OCR_OUT_DIR = os.path.join(WORKING_DIR, "ocr_output")
# --- END CONFIGURATION ---


def initialize_model():
    """
    Checks for GPU and loads the DeepSeek-OCR model into memory.
    """
    print("Initializing DeepSeek-OCR model...")
    if not torch.cuda.is_available():
        print("="*50)
        print("ERROR: No GPU found. This service requires a GPU to run.")
        print("Please run this script on a machine with an NVIDIA GPU.")
        print("="*50)
        raise SystemExit("CUDA not available.")

    print(f"GPU found: {torch.cuda.get_device_name(0)}")
    
    model_id = "deepseek-ai/DeepSeek-OCR"
    
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_safetensors=True,
        attn_implementation="eager"  # Required for this architecture
    ).to(dtype=torch.bfloat16, device="cuda").eval()
    
    print(f"Model loaded in {time.time()-t0:.1f}s")
    return model, tokenizer


def find_all_bulletin_images(url: str) -> list[str]:
    """
    Scrapes the webpage to find ALL images in the main content.
    Returns a list of full image URLs.
    """
    print(f"Scraping {url} for all images...")
    image_urls = []
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the main article content area
        content_area = soup.find('div', class_='entry-content')
        
        if not content_area:
            print("Could not find 'div.entry-content' in webpage.")
            return []
            
        # Find ALL image tags within that area
        image_tags = content_area.find_all('img')
        
        if not image_tags:
            print("Found content area, but no 'img' tags found inside.")
            return []

        for image_tag in image_tags:
            if image_tag.get('src'):
                image_url = image_tag['src']
                # Handle relative URLs if any
                if not image_url.startswith(('http:', 'https:')):
                    image_url = urljoin(url, image_url)
                image_urls.append(image_url)
            
        return image_urls
            
    except requests.RequestException as e:
        print(f"Error fetching webpage: {e}")
        return []


def download_image(url: str, save_path: str):
    """
    Downloads an image from a URL to a local path.
    """
    print(f"Downloading image from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Image saved to {save_path}")
    except requests.RequestException as e:
        print(f"Error downloading image: {e}")
        raise


def run_ocr_to_markdown(model, tokenizer, img_path: str, out_dir: str) -> str:
    """
    Runs the OCR model on a local image and returns the extracted Markdown.
    """
    print(f"Running OCR on {img_path}...")
    
    # Resize large images for speed (from your snippet)
    img = Image.open(img_path).convert("RGB")
    if max(img.size) > 2000:
        s = 2000 / max(img.size)
        img = img.resize((int(img.width*s), int(img.height*s)))
        img.save(img_path, optimize=True)

    # Use a prompt that requests Markdown output
    prompt = "<image>\n<|grounding|>Convert the document to markdown."
    
    # Clear the output directory
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "*")):
        os.remove(f)

    @torch.inference_mode()
    def _infer():
        t = time.time()
        res = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=img_path,
            output_path=out_dir,
            base_size=768,
            image_size=512,
            crop_mode=False,
            save_results=True,
            test_compress=False
        )
        print(f"OCR inference complete in {time.time()-t:.1f}s")
        return res

    _infer()

    # Find the saved Markdown file
    # The `infer` function saves results, including a markdown file.
    md_files = glob.glob(os.path.join(out_dir, "*.md"))
    if not md_files:
        print("OCR ran, but no Markdown file was found in the output directory.")
        # Fallback: check for a .txt file
        txt_files = glob.glob(os.path.join(out_dir, "*.txt"))
        if not txt_files:
            raise Exception("OCR failed to produce an output file.")
        output_file_path = txt_files[0]
    else:
        output_file_path = md_files[0]
        
    print(f"Reading OCR output from: {output_file_path}")
    with open(output_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return content


def upload_to_azure_blob(content: str, container_name: str, blob_name: str):
    """
    Uploads a string content to an Azure Blob Storage container.
    """
    if not AZURE_CONNECTION_STRING:
        print("Error: AZURE_STORAGE_CONNECTION_STRING environment variable not set.")
        print("Please set this environment variable to your Azure connection string.")
        raise ValueError("Azure connection string not found.")
        
    print(f"Uploading to Azure Blob Storage container '{container_name}' as '{blob_name}'...")
    
    try:
        # Initialize the BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        
        # Get a client for the specific blob
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        # Define content settings to set the MIME type
        content_settings = ContentSettings(content_type='text/markdown')
        
        # Upload the content (must be encoded to bytes)
        blob_client.upload_blob(content.encode('utf-8'), content_settings=content_settings, overwrite=True)
        
        print("Upload Successful.")
    
    except (AzureError, ValueError) as e:
        print(f"Error uploading to Azure Blob Storage: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during blob upload: {e}")
        raise


def get_processed_urls() -> set[str]:
    """Reads all previously processed URLs from the state file."""
    try:
        with open(STATE_FILE, 'r') as f:
            # Read all lines, strip whitespace/newlines, and return as a set
            return {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        print("No state file found. Will create a new one.")
        return set()


def add_processed_url(url: str):
    """Appends a new successfully processed URL to the state file."""
    try:
        with open(STATE_FILE, 'a') as f:
            f.write(f"{url}\n")
    except IOError as e:
        print(f"Error writing to state file {STATE_FILE}: {e}")


def main_loop():
    """
    The main service loop.
    """
    print("--- Starting Bulletin OCR Service ---")
    
    # 1. Create working directory
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(OCR_OUT_DIR, exist_ok=True)
    
    # 2. Load the model (this is the slow part)
    model, tokenizer = initialize_model()
    
    # 3. Start the loop
    while True:
        print(f"\n--- New check at {time.ctime()} ---")
        try:
            # 4. Find ALL images on the website
            current_image_urls = find_all_bulletin_images(WEBPAGE_URL)
            if not current_image_urls:
                print("Could not find any image URLs. Will try again later.")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
                
            # 5. Compare with the set of all processed images
            processed_urls = get_processed_urls()
            
            # Determine which images are new
            new_images_to_process = [
                url for url in current_image_urls if url not in processed_urls
            ]
            
            if not new_images_to_process:
                print("No new bulletin images found. Waiting...")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            print(f"Found {len(new_images_to_process)} new bulletin image(s).")
            
            # 6. Process each new image
            for image_url in new_images_to_process:
                print(f"--- Processing new image: {image_url} ---")
                try:
                    download_image(image_url, LOCAL_IMG_PATH)
                    
                    markdown_content = run_ocr_to_markdown(
                        model, tokenizer, LOCAL_IMG_PATH, OCR_OUT_DIR
                    )
                    
                    # 7. Upload to Azure Blob Storage
                    # Create a unique name (blob name)
                    blob_name = f"bulletins/{time.strftime('%Y-%m-%d_%H-%M-%S')}_bulletin.md"
                    upload_to_azure_blob(markdown_content, AZURE_CONTAINER_NAME, blob_name)
                    
                    # 8. Save state *only* for this successfully processed image
                    add_processed_url(image_url)
                    print(f"Successfully processed and uploaded: {image_url}")
                
                except Exception as e:
                    print(f"Failed to process image {image_url}: {e}")
                    # Log the error but continue to the next image
                    # The main loop's exception handler will catch fatal errors
            
            print("Finished processing all new images.")

        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
            print("Will retry after a shorter delay (60s).")
            time.sleep(60) # Short delay on error
            
        else:
            # Wait for the next poll interval if successful
            print(f"Sleeping for {POLL_INTERVAL_SECONDS} seconds...")
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    if AZURE_CONTAINER_NAME == "your-azure-container-name-here":
        print("ERROR: Please edit `bulletin_ocr_service.py` and set `AZURE_CONTAINER_NAME`.")
    elif not AZURE_CONNECTION_STRING:
        print("ERROR: AZURE_STORAGE_CONNECTION_STRING environment variable not set.")
        print("Please set this variable with your connection string before running.")
    else:
        main_loop()