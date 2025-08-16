# services/llm_planner.py
import os
import google.generativeai as genai
import logging
from dotenv import load_dotenv
from string import Template
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# --- Configuration ---
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
except ValueError as e:
    logger.error(f"API Key configuration error: {e}")

PROMPT_TEMPLATE = """
# IDENTITY AND GOAL
You are a world-class, autonomous Data Analyst Agent.
Your goal is to generate a single, correct, and executable Python script to answer the user's request.
Your entire response must be ONLY Python code, formatted in a single code block. Do not add any explanation.

---
---
# CRITICAL OUTPUT FORMAT
The final output printed to the console MUST be a JSON **array** (a list) containing the answers in the same order as the questions were asked.
- For text or numeric answers, the value should be the answer itself (e.g., "Titanic" or 0.485782).
- For plots, the value must be a base64 encoded data URI string.
- Example final output: `print(json.dumps(["Titanic", 0.485782, "data:image/png;base64,..."]))`

---
# SCRIPT INPUT MECHANISM
CRITICAL: The generated script will be run non-interactively. It MUST NOT use the `input()` function, as this will cause an `EOFError`.
The user's task (including source, questions, and output format) is passed as a JSON string via the `USER_TASK_JSON` environment variable.

Your script MUST begin with the following code to safely read and parse this input:
```python
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

task_json_string = os.getenv("USER_TASK_JSON")
if not task_json_string:
    logger.error("USER_TASK_JSON environment variable not set")
    error_payload = {{"error": "USER_TASK_JSON environment variable not set."}}
    print(json.dumps(error_payload))
    exit()

# Parse the JSON string into a Python dictionary
try:
    tasks = json.loads(task_json_string)
except Exception as e:
    logger.error(f"Failed to parse USER_TASK_JSON: {{str(e)}}")
    print(json.dumps({{"error": "Invalid USER_TASK_JSON format"}}))
    exit()

# The full task description (source, questions, output format) is in the "task" key
task_text = tasks.get("task", "")
if not task_text:
    logger.error("No task description found in USER_TASK_JSON")
    print(json.dumps({{"error": "No task description found"}}))
    exit()
```
# INPUT PARSING
# The agent must be able to handle both structured JSON and unstructured plain text.
# It MUST first attempt to find a JSON block. If and ONLY IF that fails, it should fallback to parsing a numbered list.

- **Priority 1: JSON Parsing**:
  - First, attempt to find a JSON block using `re.findall(r'{[\\s\\S]*?}', task_text)`.
  - Identify the correct block by checking if its keys are natural language questions (contain a '?').
  - If a valid question dictionary is found, extract its keys. The result of this step should be a LIST of question strings.

- **Priority 2: Plain Text Fallback**:
  - If NO valid JSON block is found, the script must assume the input is plain text.
  - It must then use a regular expression to find all questions in a numbered list format (e.g., "1. ...?", "2. ...?").
  - The result of this step should also be a LIST of question strings.

- **Final Result**: The variable holding the questions (e.g., `ordered_questions`) will be a Python list of strings, regardless of which parsing method was used.

- **Example Logic**:
  ```python
  import re
  import json

  ordered_questions = []
  
  # --- Priority 1: Attempt to parse JSON ---
  json_matches = re.findall(r'{[\\s\\S]*?}', task_text)
  if json_matches:
      for match in json_matches:
          try:
              parsed_json = json.loads(match)
              # Heuristic: Check for a dictionary of questions
              if 'questions' in parsed_json and isinstance(parsed_json['questions'], dict):
                  if any('?' in key for key in parsed_json['questions'].keys()):
                      ordered_questions = list(parsed_json['questions'].keys())
                      break # Found the correct block
          except json.JSONDecodeError:
              continue # Ignore invalid JSON blocks

  # --- Priority 2: Fallback to plain text numbered list if no JSON was found ---
  if not ordered_questions:
      # This regex finds lines starting with a number, a period, and a space.
      ordered_questions = re.findall(r"^\\s*\\d+\\.\\s*(.*?)\\?*\\s*$$", task_text, re.MULTILINE)

  if not ordered_questions:
      # If neither method works, then exit.
      print(json.dumps({"error": "Could not find a valid JSON block or a numbered list of questions."}))
      exit()
# DATA SOURCE HANDLING
Determine the data source from `task_text` and {file_names}, then load data:
- **Web (URL)**:
  - For tables: Use `pd.read_html(url)` to get all tables. Select the correct DataFrame by checking column headers (e.g., contains 'Rank', 'Title', 'Worldwide gross', 'Year', 'Peak') and row count (e.g., len(df) > 1). If no table matches, output {{"error": "No suitable table found"}}.
  - For text: Use `langchain_community.document_loaders.WebBaseLoader(url)`, parse with `BeautifulSoup`, remove `<nav>`, `<footer>`, `<script>`, `<style>`, and extract clean text.
- **S3/DuckDB**:
  - Use `duckdb.connect()`, then:
```python
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL parquet; LOAD parquet;")
    df = con.sql("SELECT * FROM read_parquet('s3://path?s3_region=ap-south-1')").df()
```
  - Handle credentials if provided in env (e.g., AWS_ACCESS_KEY_ID).
- **Attached Files**:
  - Files are in `/workspace/` with names from {file_names} (e.g., `/workspace/data.csv`).
  - CSV: `pd.read_csv('/workspace/data.csv')`
  - Images: `PIL.Image.open('/workspace/image.png')` (e.g., for description or OCR with `pytesseract`).
  - Others: Handle based on extension (e.g., `.json` with `json.load`).
- **No Source**: If task_text implies general knowledge or calculation, proceed without external data.

# METHODOLOGY
# The agent must first determine the user's primary intent and then follow the appropriate workflow.
# The agent must first determine the user's primary intent and then follow the appropriate workflow.
#Follow this workflow based on the question type:
1. **Table-Based/Math/Visualization**:ss
   - **Load**: Use appropriate method (pd.read_html, DuckDB, pd.read_csv).
   - **Clean**: For numeric columns like 'Worldwide gross':
     - - Remove footnotes with `re.sub(r'\\[.*?]', '', s)`, symbols (e.g., '$$$$', ','), 'est.', and any non-numeric prefixes...
     - Check for 'billion' or 'million' in the string before conversion (e.g., if 'billion' in s.lower(): pd.to_numeric(s.replace('billion', ''), errors='coerce') * 1e9; elif 'million' in s.lower(): pd.to_numeric(s.replace('million', ''), errors='coerce') * 1e6; else: pd.to_numeric(s, errors='coerce')).
     - Log any conversion errors with `logger.error(f"Failed to convert Worldwide gross value: {{s}}")`.
     - Use `pd.to_numeric(errors='coerce')` to convert to float, handling invalid values as NaN.
     - Drop rows with NaN in critical columns ('Worldwide gross', 'Year', 'Rank', 'Peak') before analysis. If DataFrame is empty after dropna, output {{"error": "No valid data after cleaning"}}.
     - Verify column names before accessing (e.g., use 'Title' for Wikipedia movie data). Check available columns with `df.columns` and output {{"error": "Column 'X' not found"}} if missing.
   - **Analyze**: Use pandas for filtering/grouping, scipy.stats.pearsonr for correlations, statsmodels.OLS for regressions.
   - **Visualize**: Use matplotlib/seaborn for plots. Save plots as base64 strings:
     ```python
     import io, base64
     plt.savefig(buf := io.BytesIO(), format='png', dpi=80, bbox_inches='tight')
     buf.seek(0)
     plot_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
     plt.close()
     ```
     Ensure base64 string <100,000 bytes.
2. **Text-Based (RAG)**:
   - **Load**: Use `langchain_community.document_loaders.WebBaseLoader(url)` to get documents.
   - **Clean**: Parse the document content with `BeautifulSoup`, remove `<nav>`, `<footer>`, `<script>`, and `<style>` tags, then extract the clean text.
   - **Split**: Use `langchain.text_splitter.RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` to create text chunks.
   - **Step 1: Create the Vector Database**: First, create the knowledge base. Embed the text chunks and store them in a FAISS database. The code MUST be:
     ```python
     from langchain_community.vectorstores import FAISS
     embeddings = langchain_community.embeddings.HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
     db = FAISS.from_texts(texts, embedding=embeddings)
     ```
   - **Step 2: Create the QA Chain**: Second, create the question-answering tool that USES the database. The code MUST be:
     ```python
     import langchain
     from langchain.chains import RetrievalQA
     llm = langchain_google_genai.ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=os.getenv("GOOGLE_API_KEY"), convert_system_message_to_human=True)
     qa_chain = langchain.chains.RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
     ```
   - **Step 3: Run the Chain**: Isolate the user's question from the input text and run the chain with it.
3. **Image-Based**:
   - Load with `PIL.Image.open`.
   - Describe (use pixel stats, shapes) or OCR with `pytesseract.image_to_string`.


4.  **Text-Based (Summarization)**:
   - **Use Case**: Trigger this workflow if the user asks to "summarize", "give an overview", "condense", or "explain" a document.
   - **CRITICAL - Dynamic Length Handling**: The generated script MUST intelligently handle length constraints from the user's request.
     - **Parse Request**: The script must use the `re` module to search the `task_text` for patterns like "(\\d+)\\s*words" or "(\\d+)\\s*characters".
     - **Build Dynamic Prompt**: If a limit is found, the script must create a custom `PromptTemplate` for the `load_summarize_chain` that includes the user's specific request (e.g., "Write a summary in about 150 words:"). If no limit is found, it should use a generic "Write a concise summary:" prompt.
     - **Handle Output**: If a hard character limit was requested, the script should slice the final summary string to that length (e.g., `summary[:limit]`). For word limits, the script should trust the LLM's output from the dynamic prompt.
   - **Example of the final generated code's logic**:
     ```python
     import re
     from langchain.chains.summarize import load_summarize_chain
     from langchain.prompts import PromptTemplate

     # ... (load and clean text) ...
     # ... (split text into `docs`) ...

     # --- Start of dynamic logic ---
     limit = None
     unit = "words"
     word_match = re.search(r"(\\d+)\\s*words", task_text, re.IGNORECASE)
     char_match = re.search(r"(\\d+)\\s*characters", task_text, re.IGNORECASE)

     prompt_instruction = "Write a concise summary of the following text:"
     if word_match:
         limit = int(word_match.group(1))
         prompt_instruction = f"Write a concise summary of the following text in about {{limit}} words:"
     elif char_match:
         limit = int(char_match.group(1))
         unit = "characters"
         prompt_instruction = f"Write a concise summary of the following text, keeping it under {{limit}} characters:"

     prompt_template = f"{{prompt_instruction}}\\n\\n\"{{text}}\"\\n\\nCONCISE SUMMARY:"
     PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
     docs = [Document(page_content=cleaned_text)]
     llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=os.getenv("GOOGLE_API_KEY"))
     summary_chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
     summary = summary_chain.run(docs)
     if unit == "characters" and limit is not None:
         final_answer = summary[:limit]
     else:
         final_answer = summary
     # --- End of dynamic logic ---
     ```
     
5. S3/DuckDB-Based Analysis
# Use this for tasks involving querying large Parquet datasets from S3 using DuckDB.
 **CRITICAL**: To handle large datasets efficiently, the script MUST apply filters directly in the DuckDB SQL query using a `WHERE` clause whenever possible. Do not load the entire dataset with `SELECT *` if you only need a subset. Subsequent analysis can then be done on the much smaller, pre-filtered pandas DataFrame.
- **Dynamic S3 Path**: The generated script MUST first use the `re` module to find and extract the S3 path (which starts with `s3://`) from the `task_text`.
- **Data Loading**: After extracting the path, the script must connect to DuckDB and query the S3 path. It should handle cases where no path is found.
- **Analysis**: All subsequent analysis (filtering, cleaning, regression) should be done on the loaded pandas DataFrame.
- **Example of the generated code's logic**:
```python
# Example of the generated code's logic:
  ```python
  import os
  import json
  import logging
  import re
  import pandas as pd
  import duckdb
  import matplotlib.pyplot as plt
  import seaborn as sns
  import base64
  import io
  from scipy.stats import linregress

  # --- Boilerplate Setup ---
  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger(__name__)

  task_json_string = os.getenv("USER_TASK_JSON")
  # ... (rest of boilerplate setup) ...
  
  # --- Dynamic Input Parsing ---
  question_dict = None
  # ... (rest of input parsing) ...

  s3_match = re.search(r"s3://[^\\s']+", task_text)
  s3_path = s3_match.group(0) if s3_match else None

  # --- Ordered Question Answering ---
  # Initialize an empty LIST to store answers in order
  final_answers = []
  
  # Establish the exact order of questions from the parsed JSON
  ordered_questions = list(question_dict.keys())

  # Reusable connection for efficiency
  con = duckdb.connect(config={'s3_region': 'ap-south-1'})
  con.execute("INSTALL httpfs; LOAD httpfs; INSTALL parquet; LOAD parquet;")
  
  # Use a shared DataFrame for the regression and plot to avoid fetching data twice
  regression_df = None

  for question in ordered_questions:
      question_lower = question.lower()
      
      try:
          # **DYNAMIC LOGIC: Check for keywords to decide which analysis to run**
          if 'most cases' in question_lower:
              logger.info(f"Answering question: {question}")
              query = f"SELECT court FROM read_parquet('{s3_path}') WHERE year BETWEEN 2019 AND 2022"
              df = con.sql(query).df()
              answer = df['court'].mode()[0]
              final_answers.append(answer)

          elif 'regression slope' in question_lower:
              logger.info(f"Answering question: {question}")
              query = f"SELECT decision_date, date_of_registration, year FROM read_parquet('{s3_path}') WHERE court = '33_10'"
              df = con.sql(query).df()
              
              df['decision_date'] = pd.to_datetime(df['decision_date'], errors='coerce')
              df['date_of_registration'] = pd.to_datetime(df['date_of_registration'], errors='coerce')
              df['delay_days'] = (df['decision_date'] - df['date_of_registration']).dt.days
              regression_df = df.dropna(subset=['year', 'delay_days'])
              
              slope, _, _, _, _ = linregress(regression_df['year'], regression_df['delay_days'])
              final_answers.append(slope)

          elif 'plot' in question_lower and 'scatterplot' in question_lower:
              logger.info(f"Answering question: {question}")
              if regression_df is None:
                  logger.info("Plot data not found, fetching...")
                  query = f"SELECT decision_date, date_of_registration, year FROM read_parquet('{s3_path}') WHERE court = '33_10'"
                  df = con.sql(query).df()
                  df['decision_date'] = pd.to_datetime(df['decision_date'], errors='coerce')
                  df['date_of_registration'] = pd.to_datetime(df['date_of_registration'], errors='coerce')
                  df['delay_days'] = (df['decision_date'] - df['date_of_registration']).dt.days
                  regression_df = df.dropna(subset=['year', 'delay_days'])
              
              plt.figure(figsize=(8, 6))
              sns.regplot(x='year', y='delay_days', data=regression_df)
              plt.title('Delay vs. Year (Court 33_10)')
              buf = io.BytesIO()
              plt.savefig(buf, format='webp', dpi=80, bbox_inches='tight')
              buf.seek(0)
              plot_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
              plt.close()
              final_answers.append(f"data:image/webp;base64,{plot_b64}")
          else:
              # If the question is not recognized, append an error
              final_answers.append(f"Error: Unrecognized question format '{question}'")

      except Exception as e:
          logger.error(f"Error processing question '{question}': {str(e)}")
          # Append an error message to the list to maintain order
          final_answers.append(f"Error: Could not answer question.")

  con.close()
  # The final print MUST be a JSON array (list)
  print(json.dumps(final_answers, ensure_ascii=False))
```python
# --- END OF DYNAMIC LOGIC ---
  # For plotting, use matplotlib and seaborn on the pandas DataFrame
# CRITICAL RULES & CONSTRAINTS
- - **Error Handling**: If the entire script fails (e.g., cannot load data), output a single JSON object like {{"error": "Descriptive error message"}}. For multi-question tasks, if a single question fails, its corresponding entry in the final JSON array/object **MUST** contain the error message. **Do not use empty strings for failures.** This ensures partial credit can be awarded and debugging is possible. Always log errors with logger.error.
- **Output Format**: Match task_text (e.g., JSON array for numbered questions, object for key-value). Always print: `print(json.dumps(final_answers, ensure_ascii=False))`.
- **Plot Size**: Base64-encoded plots MUST be <100,000 bytes (use PNG, dpi=80, bbox_inches='tight').
- **Non-Interactive**: No `input()`. All inputs via USER_TASK_JSON or /workspace files.
- **Dependencies**: Use only: pandas, numpy, scipy, matplotlib, seaborn, beautifulsoup4, requests, lxml, duckdb, pillow, scikit-learn, statsmodels, langchain-community, sentence-transformers, faiss-cpu, torch, pytesseract, google-generativeai, langchain-google-genai.
# FINAL ASSEMBLY & OUTPUT
- Initialize a list or dictionary to hold the answers.
- **CRITICAL**: For tasks with multiple questions, wrap the logic for EACH question in its own `try...except` block.
- If a question's logic succeeds, store the result.
- If a question's logic fails, catch the exception, log it, and store a clear error message as a string (e.g., `"Error: Could not calculate correlation."`) in that question's answer slot.
- This guarantees that a failure in one question does not prevent others from being answered.
- Always print the final JSON result. `print(json.dumps(final_answers, ensure_ascii=False))`.

# USER TASK
Task Description:
$task_description

Available Files:
$file_names
"""

def generate_script(task_description: str, file_names: list[str]) -> str | None:
    """
    Generates a Python script using the Gemini model based on a task description.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("Cannot generate script because GOOGLE_API_KEY is not set.")
        return None

    model = genai.GenerativeModel(
        'gemini-1.5-pro-latest',
        safety_settings={
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
        }
    )
    
    template = Template(PROMPT_TEMPLATE)
    final_prompt = template.substitute(
        task_description=task_description,
        file_names=str(file_names) 
    )
    try:
        logger.info("Generating script with Gemini API...")
        response = model.generate_content(final_prompt)
        
        # Extract only Python code
        script_code = response.text.strip()
        if script_code.startswith("```python"):
            script_code = script_code[9:]
        if script_code.endswith("```"):
            script_code = script_code[:-3]
        
        logger.info("Successfully generated script.")
        return script_code.strip()

    except Exception as e:
        logger.error(f"An error occurred while calling the Gemini API: {e}")
        return None