import runpod
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from huggingface_hub import snapshot_download, hf_hub_download
from PIL import Image
import base64
import io
import os
import logging
import time
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline variable
pipeline = None
model_loaded = False

def get_cache_dir():
    """Determine the best cache directory to use"""
    # Check if network volume is available
    if os.path.exists('/runpod-volume'):
        cache_dir = '/runpod-volume/models'
        os.makedirs(cache_dir, exist_ok=True)
        logger.info("Using network volume for model cache")
        return cache_dir

    # Use environment variable or default
    cache_dir = os.environ.get('MODEL_CACHE_DIR', '/tmp/models')
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"Using local cache directory: {cache_dir}")
    return cache_dir

def download_model_if_needed(model_id, cache_dir):
    """Download model only if not already cached"""
    model_path = os.path.join(cache_dir, model_id.replace('/', '--'))

    # Check if model is already downloaded
    if os.path.exists(model_path) and os.listdir(model_path):
        logger.info(f"Model {model_id} already cached at {model_path}")
        return model_path

    logger.info(f"Downloading model {model_id}...")
    start_time = time.time()

    try:
        # Download model to cache
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_files_only=False,
            resume_download=True
        )

        download_time = time.time() - start_time
        logger.info(f"Model downloaded in {download_time:.1f}s")
        return model_path

    except Exception as e:
        logger.error(f"Failed to download model {model_id}: {e}")
        raise

def load_model():
    """Load the model with smart caching"""
    global pipeline, model_loaded

    if model_loaded:
        return True

    model_id = os.environ.get('MODEL_ID', 'runwayml/stable-diffusion-v1-5')
    cache_dir = get_cache_dir()

    try:
        logger.info(f"Loading model: {model_id}")

        # Download if needed
        download_model_if_needed(model_id, cache_dir)

        # Load pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True,
            local_files_only=False  # Allow download if needed
        )

        # Optimize pipeline
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        pipeline = pipeline.to("cuda")
        pipeline.enable_attention_slicing()

        # Enable memory efficient attention if available
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            logger.info("xformers memory efficient attention enabled")
        except Exception as e:
            logger.warning(f"xformers not available: {e}")

        # Enable model CPU offload for VRAM optimization
        try:
            pipeline.enable_model_cpu_offload()
            logger.info("Model CPU offload enabled")
        except Exception as e:
            logger.warning(f"Model CPU offload failed: {e}")

        model_loaded = True
        logger.info("Model loaded successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

def cleanup_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def validate_inputs(job_input):
    """Validate and sanitize input parameters"""
    prompt = job_input.get('prompt', 'a beautiful landscape')
    negative_prompt = job_input.get('negative_prompt', '')
    width = min(max(job_input.get('width', 512), 256), 1024)
    height = min(max(job_input.get('height', 512), 256), 1024)
    num_inference_steps = min(max(job_input.get('steps', 20), 10), 100)
    guidance_scale = min(max(job_input.get('guidance_scale', 7.5), 1.0), 20.0)
    seed = job_input.get('seed', None)
    num_images = min(max(job_input.get('num_images', 1), 1), 4)

    # Ensure dimensions are multiples of 8 (required for stable diffusion)
    width = (width // 8) * 8
    height = (height // 8) * 8

    return {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'width': width,
        'height': height,
        'num_inference_steps': num_inference_steps,
        'guidance_scale': guidance_scale,
        'seed': seed,
        'num_images': num_images
    }

def handler(job):
    """Main handler function for RunPod serverless"""
    global pipeline

    start_time = time.time()

    # Load model if not already loaded
    if not model_loaded:
        logger.info("Model not loaded, loading now...")
        if not load_model():
            return {
                "status": "error",
                "error": "Failed to load model",
                "generation_time": time.time() - start_time
            }

    # Get and validate job input
    job_input = job.get('input', {})
    params = validate_inputs(job_input)

    try:
        # Set seed if provided
        generator = None
        if params['seed'] is not None:
            generator = torch.Generator(device="cuda").manual_seed(params['seed'])

        # Generate images
        logger.info(f"Generating {params['num_images']} image(s): {params['prompt'][:50]}...")
        generation_start = time.time()

        with torch.autocast("cuda"):
            result = pipeline(
                prompt=params['prompt'],
                negative_prompt=params['negative_prompt'] if params['negative_prompt'] else None,
                width=params['width'],
                height=params['height'],
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                num_images_per_prompt=params['num_images'],
                generator=generator
            )

        generation_time = time.time() - generation_start

        # Convert images to base64
        images_b64 = []
        for image in result.images:
            images_b64.append(image_to_base64(image))

        # Clean up memory
        cleanup_memory()

        total_time = time.time() - start_time
        logger.info(f"Total request completed in {total_time:.2f}s (generation: {generation_time:.2f}s)")

        return {
            "status": "success",
            "images": images_b64,
            "prompt": params['prompt'],
            "model_id": os.environ.get('MODEL_ID', 'unknown'),
            "generation_time": round(generation_time, 2),
            "total_time": round(total_time, 2),
            "parameters": params
        }

    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        cleanup_memory()
        return {
            "status": "error",
            "error": str(e),
            "generation_time": time.time() - start_time
        }

# Pre-load model when container starts (but don't fail if it fails)
if __name__ == "__main__":
    logger.info("Starting RunPod serverless worker...")

    # Try to pre-load model
    try:
        load_model()
        logger.info("Model pre-loaded successfully")
    except Exception as e:
        logger.warning(f"Model pre-load failed, will load on first request: {e}")

    # Start the serverless worker
    runpod.serverless.start({"handler": handler})
