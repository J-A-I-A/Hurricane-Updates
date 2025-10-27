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
from azure.core.exceptions import AzureError, ResourceNotFoundError

# --- RunPod Import ---
import runpod

# --- CONFIGURATION ---
# These are now loaded from environment variables (set in RunPod Secrets)
AZURE_CONTAINER_NAME = os.environ.get("AZURE_CONTAINER_NAME")
AZURE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

# The webpage to scrape (can be overridden by handler input)
WEBPAGE_URL = "https://www.odpem.org.jm/weather-alert-melissa/"

# Local storage setup (ephemeral, just for this run)
WORKING_DIR = "/tmp/bulletin_ocr"
STATE_FILE_NAME = "bulletin_processed_urls.txt" # The name of the state file *in blob storage*
LOCAL_IMG_PATH = os.path.join(WORKING_DIR, "bulletin_image.jpg")
OCR_OUT_DIR = os.path.join(WORKING_DIR, "ocr_output")
# --- END CONFIGURATION ---


def initialize_model():
    """
    Checks for GPU and loads the DeepSeek-OCR model into memory.
    This runs ONCE when the RunPod worker starts.
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


def download_image(url: str, save_path: str):
    """
    Downloads an image from a URL to a local path.
    """
    # ... (This function is unchanged from your version) ...
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


@torch.inference_mode()
def run_ocr_model(model, tokenizer, img_path: str) -> str:
    """
    Runs the DeepSeek-OCR model on a local image file and returns the
    extracted text as a markdown string.
    """
    print(f"Starting OCR for {img_path}...")
    t = time.time()

    # The prompt to instruct the model to return markdown
    prompt = "<image>\n<|grounding|>Convert the document to markdown."
    
    try:
        # Run inference
        # We set save_results=False as we don't need the files, just the text.
        res = model.infer(
            tok,
            prompt=prompt,
            image_file=img_path,
            output_path=None,      # No output path
            base_size=768,       
            image_size=512,      
            crop_mode=False,     
            save_results=False,  # <-- IMPORTANT: Do not save to disk
            test_compress=False
        )
        
        print(f"OCR inference complete in {time.time()-t:.1f}s")
        
        # The 'res' object is a string containing the markdown output
        markdown_text = res
        print("YOWWW")
        print(markdown_text)
        
        if not markdown_text or markdown_text.isspace():
            raise ValueError("OCR ran but returned no text.")
            
        return markdown_text

    except Exception as e:
        print(f"Error during OCR inference: {e}")
        # Re-raise to be caught by the main handler
        raise


# def run_ocr_model(model, tokenizer, img_path: str, out_dir: str) -> str:
#     """
#     Runs the OCR model on a local image and returns the extracted Markdown.
#     """
#     # ... (This function is unchanged from your version) ...
#     print(f"Running OCR on {img_path}...")
    
#     # Resize large images for speed (from your snippet)
#     img = Image.open(img_path).convert("RGB")
#     if max(img.size) > 2000:
#         s = 2000 / max(img.size)
#         img = img.resize((int(img.width*s), int(img.height*s)))
#         img.save(img_path, optimize=True)

#     # Use a prompt that requests Markdown output
#     prompt = "<image>\n<|grounding|>Convert the document to markdown."
    

#     @torch.inference_mode()
#     def _infer():
#         t = time.time()
#         res = model.infer(
#             tokenizer,
#             prompt=prompt,
#             image_file=img_path,
#             output_path=None,
#             base_size=768,
#             image_size=512,
#             crop_mode=False,
#             save_results=False,
#             test_compress=False
#         )
#         print(f"OCR inference complete in {time.time()-t:.1f}s")
#         return res

#     markdown_text = _infer()

#     # The 'res' object is a string containing the markdown output
#     markdown_text = res
        
#     if not markdown_text or markdown_text.isspace():
#         raise ValueError("OCR ran but returned no text.")
            
#     return markdown_text


def upload_to_azure_blob(content: str | bytes, container_name: str, blob_name: str, content_type: str):
    """
    Uploads string or bytes content to an Azure Blob Storage container.
    """
    if not AZURE_CONNECTION_STRING:
        raise ValueError("Azure connection string not found.")
        
    print(f"Uploading to Azure Blob Storage container '{container_name}' as '{blob_name}'...")
    
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        content_settings = ContentSettings(content_type=content_type)
        
        # Encode string to bytes if needed
        if isinstance(content, str):
            content_bytes = content.encode('utf-8')
        else:
            content_bytes = content

        blob_client.upload_blob(content_bytes, content_settings=content_settings, overwrite=True)
        print("Upload Successful.")
    
    except (AzureError, ValueError) as e:
        print(f"Error uploading to Azure Blob Storage: {e}")
        raise


def get_processed_urls(container_name: str, state_blob_name: str) -> set[str]:
    """Reads all previously processed URLs from the state file in Blob Storage."""
    if not AZURE_CONNECTION_STRING:
        raise ValueError("Azure connection string not found.")
        
    print(f"Fetching processed URL list from blob: {state_blob_name}")
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=state_blob_name)
        
        # Download blob content
        blob_data = blob_client.download_blob().readall()
        content = blob_data.decode('utf-8')
        
        # Read all lines, strip whitespace/newlines, and return as a set
        return {line.strip() for line in content.splitlines() if line.strip()}
        
    except ResourceNotFoundError:
        print("No state file blob found. Will create a new one.")
        return set()
    except (AzureError, ValueError) as e:
        print(f"Error downloading state file: {e}")
        raise


def save_processed_urls(url_set: set[str], container_name: str, state_blob_name: str):
    """Saves the complete set of processed URLs back to Blob Storage, overwriting."""
    print(f"Saving updated URL list to blob: {state_blob_name}")
    # Join the set into a single string, one URL per line
    content = "\n".join(sorted(list(url_set)))
    upload_to_azure_blob(content, container_name, state_blob_name, content_type='text/plain')


# --- Load Model Globally ---
# This runs when the RunPod worker boots up, *before* the handler.
# This keeps the model "hot" in memory.
model, tokenizer = initialize_model()


# --- RunPod Handler ---
def handler(event):
    """
    This function is called by RunPod for each HTTP request.
    """
    print(f"Handler started at {time.ctime()}")
    
    # 1. Get config (from env vars or override via event)
    config = event.get("input", {})
    container_name = config.get("container_name", AZURE_CONTAINER_NAME)
    scrape_url = config.get("scrape_url", WEBPAGE_URL)
    
    if not container_name or not AZURE_CONNECTION_STRING:
        return {"error": "AZURE_CONTAINER_NAME or AZURE_STORAGE_CONNECTION_STRING not set."}

    # 2. Create local working directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(OCR_OUT_DIR, exist_ok=True)
    
    processed_urls = set()
    new_images_count = 0
    
    try:
        # 3. Find ALL images on the website
        current_image_urls = find_all_bulletin_images(scrape_url)
        if not current_image_urls:
            return {"status": "Complete", "message": "Could not find any image URLs."}
            
        # 4. Compare with the set of all processed images from Azure Blob
        processed_urls = get_processed_urls(container_name, STATE_FILE_NAME)
        
        new_images_to_process = [
            url for url in current_image_urls if url not in processed_urls
        ]
        
        if not new_images_to_process:
            return {"status": "Complete", "message": "No new bulletin images found."}

        print(f"Found {len(new_images_to_process)} new bulletin image(s).")
        
        # 5. Process each new image
        for image_url in new_images_to_process:
            print(f"--- Processing new image: {image_url} ---")
            try:
                download_image(image_url, LOCAL_IMG_PATH)
                
                markdown_content = run_ocr_model(
                    model, tokenizer, LOCAL_IMG_PATH, OCR_OUT_DIR
                )
                
                # Create a unique blob name
                blob_name = f"bulletins/{time.strftime('%Y-%m-%d_%H-%M-%S')}_{os.path.basename(image_url)}.md"
                upload_to_azure_blob(markdown_content, container_name, blob_name, content_type='text/markdown')
                
                # Add to our set for final state save
                processed_urls.add(image_url)
                new_images_count += 1
                print(f"Successfully processed and uploaded: {image_url}")
            
            except Exception as e:
                print(f"Failed to process image {image_url}: {e}")
                # Log error but continue to the next image

        # 6. Save the new complete state file
        if new_images_count > 0:
            save_processed_urls(processed_urls, container_name, STATE_FILE_NAME)
            
        return {
            "status": "Complete",
            "message": f"Finished processing. Uploaded {new_images_count} new bulletin(s)."
        }

    except Exception as e:
        print(f"An error occurred in the handler: {e}")
        # Return an error so RunPod can log it
        return {"error": str(e)}


# --- Start Serverless Worker ---
if __name__ == "__main__":
    if not AZURE_CONTAINER_NAME or not AZURE_CONNECTION_STRING:
        print("="*50)
        print("ERROR: AZURE_CONTAINER_NAME or AZURE_STORAGE_CONNECTION_STRING not set.")
        print("Please set these environment variables before running.")
        print("="*50)
    else:
        print("Starting RunPod serverless worker...")
        runpod.serverless.start({"handler": handler})