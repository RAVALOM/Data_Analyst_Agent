import os
import shutil
import uuid
import json
from pathlib import Path
import asyncio
import logging
import re  # Import the regular expression module
from fastapi import FastAPI, Request, HTTPException, Response
from dotenv import load_dotenv
import uvicorn

# Import custom modules
from services.llm_planner import generate_script
from agent.executor import run_script_in_sandbox

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initial Setup ---
# Load environment variables from .env file
load_dotenv()

# Check for API Key
if not os.getenv("GOOGLE_API_KEY"):
    logger.error("FATAL ERROR: GOOGLE_API_KEY environment variable is not set.")
    exit(1)

# Create a 'temp' directory if it doesn't exist
Path("temp").mkdir(exist_ok=True)

# Initialize the FastAPI app
app = FastAPI(
    title="D    ata Analyst Agent API",
    description="An API that uses LLMs to source, prepare, analyze, and visualize data.",
    version="1.0.0"
)

# --- API Endpoint ---
@app.post("/api/")
async def analyze_data(request: Request):
    """
    Receives a data analysis task, generates a Python script, executes it in a sandbox, and returns the result.
    """
    temp_dir = Path("temp") / str(uuid.uuid4())
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Set timeout for entire request (180s per project requirement)
        async with asyncio.timeout(180):
            form_data = await request.form()

            if "questions.txt" not in form_data:
                raise HTTPException(
                    status_code=400,
                    detail="A 'questions.txt' file is required in the form data."
                )

            full_task_description = ""
            uploaded_file_names = []

            # Process uploaded files
            for name, file in form_data.items():
                if not file.filename:
                    continue
                
                file_path = temp_dir / Path(file.filename).name
                
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(file.file, f)

                if name == "questions.txt":
                    full_task_description = file_path.read_text()
                
                uploaded_file_names.append(file.filename)

            if not full_task_description:
                 raise HTTPException(status_code=400, detail="questions.txt is empty or was not provided.")

            logger.info(f"Uploaded files: {uploaded_file_names}")
            
            # --- START OF MODIFICATION ---
            # Make the task description robust by extracting the JSON part if it exists.
            # This helps the LLM focus on the core questions while allowing for other formats.
            logger.info("Parsing task description to find a JSON object.")
            json_match = re.search(r'\{[\s\S]*\}', full_task_description)
            task_description = full_task_description # Default to the full text

            if json_match:
                logger.info("JSON object found in task description. Attempting to use it as the primary task.")
                try:
                    # We found a JSON block, assume this is the core task.
                    # We'll re-parse it to ensure it's valid and to pretty-print for the LLM.
                    parsed_json = json.loads(json_match.group(0))
                    # The task for the LLM and sandbox is now just the clean JSON.
                    task_description = json.dumps(parsed_json, indent=2)
                    logger.info("Successfully parsed JSON block. Using it as the task.")
                except json.JSONDecodeError:
                    # If parsing fails, it's not a valid JSON object.
                    # We'll log a warning and use the original full text.
                    logger.warning("Found a JSON-like block, but it failed to parse. Using full description.")
                    task_description = full_task_description
            else:
                logger.info("No JSON object found. Using the full text as the task description.")
            # --- END OF MODIFICATION ---
            
            # Generate script using the (potentially cleaned) task description
            script_code = generate_script(task_description, uploaded_file_names)
            if not script_code:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to generate analysis script from LLM."
                )
            
            logger.debug(f"Generated script snippet: {script_code[:200]}...")

            # Pass the (potentially cleaned) task description in USER_TASK_JSON
            task_payload = {"task": task_description}
            task_json_string = json.dumps(task_payload)

            script_env = {
                "USER_TASK_JSON": task_json_string,
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
                "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
                "HUGGINGFACEHUB_API_TOKEN": os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
            }

            output, error = run_script_in_sandbox(script_code, temp_dir, script_env)

            if error:
                raise HTTPException(status_code=500, detail=f"Script execution failed: {error}")
            
            try:
                json.loads(output)
            except (json.JSONDecodeError, TypeError) as e:
                raise HTTPException(status_code=500, detail=f"Script produced invalid JSON output:\n{output}\nError: {str(e)}")

            return Response(content=output, media_type="application/json")

    except asyncio.TimeoutError:
        logger.error("Request timed out after 180 seconds")
        raise HTTPException(status_code=504, detail="Request timed out after 180 seconds")
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

# --- Health Check Endpoint ---
@app.get("/")
def read_root():
    return {"status": "Data Analyst Agent is running"}

# --- Uvicorn Runner ---
if __name__ == "__main__":
    uvicorn.run(
        "main:app",              # Target the app instance
        host="0.0.0.0",
        port=8000,
        reload=True,             # Keep auto-reload enabled
        reload_excludes=["temp"] # Exclude the temp directory from the watcher
    )
