# In agent/executor.py

import docker
import os
import uuid
import logging
from pathlib import Path
from docker.errors import NotFound, APIError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
DOCKER_IMAGE = "data-analyst-sandbox:latest"
TIMEOUT_SECONDS = 700  # Script execution timeout (less than the 5-minute API limit)
CLIENT_TIMEOUT = 700   # Docker client connection timeout (matches the 5-minute API limit)
MODEL_CACHE_VOLUME = "huggingface_model_cache"

# --- MODIFIED FUNCTION SIGNATURE ---
def run_script_in_sandbox(script_code: str, temp_dir: Path, env_vars: dict) -> tuple[str | None, str | None]:
    """
    Executes a Python script in a secure, isolated Docker container with a shared model cache.
    """
    # --- FIX: Initialize the client with a longer timeout that respects the API limit ---
    client = docker.from_env(timeout=CLIENT_TIMEOUT)

    # Create the model cache volume if it doesn't exist
    try:
        client.volumes.get(MODEL_CACHE_VOLUME)
        logger.info(f"Volume '{MODEL_CACHE_VOLUME}' already exists.")
    except NotFound:
        try:
            client.volumes.create(name=MODEL_CACHE_VOLUME)
            logger.info(f"Created volume '{MODEL_CACHE_VOLUME}'.")
        except APIError as e:
            logger.error(f"Failed to create volume '{MODEL_CACHE_VOLUME}': {e}")
            return None, f"Failed to create volume '{MODEL_CACHE_VOLUME}': {e}"

    # Verify the Docker image exists
    try:
        client.images.get(DOCKER_IMAGE)
        logger.info(f"Docker image '{DOCKER_IMAGE}' found.")
    except docker.errors.ImageNotFound:
        logger.error(f"Docker image '{DOCKER_IMAGE}' not found.")
        return None, f"Docker image '{DOCKER_IMAGE}' not found. Please build it first."
    except APIError as e:
        logger.error(f"Failed to connect to Docker daemon: {e}")
        return None, f"Failed to connect to Docker daemon: {e}"

    # Write the script to a temporary file
    script_filename = f"script_{uuid.uuid4()}.py"
    script_path = temp_dir / script_filename
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_code)
        logger.debug(f"Wrote script to {script_path}")
    except Exception as e:
        logger.error(f"Failed to write script to {script_path}: {e}")
        return None, f"Failed to write script file: {e}"

    cache_path_in_container = "/huggingface_cache"
    
    # Merge environment variables
    final_environment_variables = {
        "SENTENCE_TRANSFORMERS_HOME": cache_path_in_container
    }
    if env_vars:
        final_environment_variables.update(env_vars)

    volume_mounts = {
        str(temp_dir.resolve()): {
            "bind": "/workspace",
            "mode": "rw"
        },
        MODEL_CACHE_VOLUME: {
            "bind": cache_path_in_container,
            "mode": "rw"
        }
    }
    
    container = None
    try:
        logger.info(f"Running script '{script_filename}' in Docker container...")
        container = client.containers.run(
            image=DOCKER_IMAGE,
            command=["python", script_filename],
            volumes=volume_mounts,
            working_dir="/workspace",
            environment=final_environment_variables,
            mem_limit="2g",  # Limit memory to 2GB
            cpu_quota=100000,  # Limit CPU usage (100ms per second)
            network_mode="bridge",  # Ensure internet access
            detach=True,
            remove=False,
        )

        result = container.wait(timeout=TIMEOUT_SECONDS)
        
        stdout = container.logs(stdout=True, stderr=False).decode("utf-8").strip()
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8").strip()

        if result["StatusCode"] == 0:
            logger.info("Script executed successfully.")
            return stdout, None
        else:
            error_msg = f"Script execution failed with status code {result['StatusCode']}:\n{stderr}"
            logger.error(error_msg)
            return None, error_msg

    except docker.errors.ContainerError as e:
        logger.error(f"ContainerError: {e.stderr.decode('utf-8')}")
        return None, e.stderr.decode('utf-8')
    except docker.errors.APIError as e:
        # This will now catch the timeout from the client itself if it still occurs
        if "Read timed out" in str(e):
             logger.error(f"Docker client connection timed out after {CLIENT_TIMEOUT} seconds.")
             return None, f"Docker client connection timed out after {CLIENT_TIMEOUT} seconds."
        logger.error(f"Docker API error: {e}")
        return None, f"Docker API error: {e}"
    except TimeoutError: # This is for container.wait()
        logger.error(f"Script execution timed out after {TIMEOUT_SECONDS} seconds")
        return None, f"Script execution timed out after {TIMEOUT_SECONDS} seconds"
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return None, f"An unexpected error occurred: {e}"
    finally:
        if container:
            try:
                container.remove(force=True)
                logger.debug(f"Removed container {container.id}")
            except docker.errors.NotFound:
                logger.debug(f"Container {container.id} already removed")
            except APIError as e:
                logger.error(f"Failed to remove container: {e}")
