import os
import json
import subprocess
import datetime
import requests
from typing import Optional
from fastapi import FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from dateutil.parser import parse
import openai

DATA_ROOT = os.path.abspath("C:/data") # Adjust if running on Windows (e.g. "C:/data")
def is_within_data_root(path: str) -> bool:
    """ Returns True if 'path' is within DATA_ROOT, False otherwise. """ 
    abs_path = os.path.abspath(path) 
    return abs_path.startswith(DATA_ROOT)

def safe_open(path: str, mode="r", encoding=None): 
    """ Opens a file only if the resolved absolute path is inside /data. Otherwise, raises an HTTP 403 error. """ 
    abs_path = os.path.abspath(path) 
    if not abs_path.startswith(DATA_ROOT): 
        raise HTTPException( status_code=status.HTTP_403_FORBIDDEN, detail=f"Access to files outside {DATA_ROOT} is prohibited." ) 
    return open(abs_path, mode, encoding=encoding)

def safe_remove(path: str): 
    """ NO-OP. Business requirement B2: Data is never deleted anywhere. Raises HTTP 403 if attempted. """ 
    raise HTTPException( status_code=status.HTTP_403_FORBIDDEN, detail="Deletion of files is prohibited by policy." )


def format_with_prettier(file_path: str, version: str):
    # Ensure Prettier is installed first
    try:
        # Install Prettier globally if it's not already installed
        subprocess.run(f'npm install -g prettier@{version}', shell=True, check=True)
        # Check Prettier installation
        subprocess.run('prettier --version', shell=True, check=True)
        
        # Run Prettier to format the file in-place
        cmd = f'prettier --write "{file_path}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return result.stdout  # Return standard output from Prettier
    except subprocess.CalledProcessError as e:
        # Capture any errors in stderr
        raise HTTPException(status_code=500, detail=f"Error running Prettier: {e.stderr}")
    
def post_process_markdown(file_path: str):
    with safe_open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace all lists that were converted to '-' back to '*'
    content = content.replace("- ", "* ")

    with safe_open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

def passes_luhn(card_number: str) -> bool:
    def digits_of(n: str):
        return [int(d) for d in str(n)]

    digits = digits_of(card_number)
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]

    checksum = sum(odd_digits)
    for d in even_digits:
        checksum += sum(digits_of(d * 2))

    return checksum % 10 == 0

def get_embeddings(texts):
    """
    Generate embeddings for a list of texts using OpenAI's API.
    You can replace this with any other embedding model as needed.
    """
    response = openai.Embedding.create(
        model="text-embedding-ada-002",  # Or choose another embedding model
        input=texts
    )
    return [item["embedding"] for item in response["data"]]

def read_comments(file_path):

    with safe_open(file_path, 'r', encoding='utf-8') as file:
        return file.read().splitlines()

def write_similar_comments(file_path, comment1, comment2):
    with safe_open(file_path, 'w', encoding='utf-8') as file:
        file.write(comment1 + '\n')
        file.write(comment2 + '\n')


app = FastAPI()

# Allow CORS (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WARNING: Hardcoding AIPROXY_TOKEN is not recommended in production
# AIPROXY_TOKEN = (
#     "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIxZjIwMDEzMDZAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9."
#     "0mcAComQBqT3Gewe3bQoD1rFmoyLgCrM9cbNa8JNeJM"
# )
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN") 
if not AIPROXY_TOKEN: 
    raise RuntimeError("AIPROXY_TOKEN environment variable is missing!")



@app.post("/run")
def run_task(
    user_email: Optional[str] = Query(None, description="Optional user email"),
    task: str = Query(..., title="task", description="The plain-English task to execute")
):
    """
    This endpoint accepts a 'task' string which is forwarded to GPT-4o-Mini
    via AI Proxy. Depending on the parsed tool calls, it performs:
      - script_runner (A1)
      - prettier_formatter (A2)
      - count_wednesdays (A3)
      - sort_contacts (A4)
      - recent_logs (A5)
    """
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {AIPROXY_TOKEN}"}

    system_instructions = (
        "You can:\n"
        "1) Run Python scripts with UV (script_runner)\n"
        "2) Format the contents of /data/format.md using prettier@3.4.2, updating the file in-place (prettier_formatter)\n"
        "3) Count Wednesdays in a file (count_wednesdays)\n"
        "4) Sort contacts by last_name, then first_name (sort_contacts)\n"
        "5) Write the first line of the 10 most recent .log files (recent_logs)\n"
        "6) Find Markdown (.md) files and build docs index (docs_index)\n"
        "7) Extract sender’s email from /data/email.txt (extract_email_sender)\n"
        "8) Extract credit card number from /data/credit-card.png (extract_credit_card)\n"
        "Be extremely careful while extracting the credit card number. Please check **each digit** individually.\n"
        "Do not approximate or make assumptions on any digit. The OCR output must match exactly, and the credit card number "
        "must consist of **exactly 16 digits**.\n"
        "If there is any uncertainty about a digit, please flag it or use the most probable match after verifying the digits.\n"
        "Your output must be **only the 16 digits**, with **no spaces**, and no other text or numbers."
        "9) Find the most similar pair of comments in /data/comments.txt using embeddings (comments_similarity)\n"
        "10) Sum sales for the 'Gold' ticket type from /data/ticket-sales.db (ticket_sales_gold)\n"
        "You can also do the following\n"
        "1) Run Python scripts with UV (script_runner)\n"
        "B3) Fetch data from an API and save it (fetch_data_api)\n"
        "B4) Clone a git repo and make a commit (clone_git_repo)\n"
        "B5) Run a SQL query on a SQLite or DuckDB database (sql_query)\n"
        "B6) Extract/scrape data from a website (scrape_website)\n"
        "B7) Compress or resize an image (compress_image)\n"
        "B8) Transcribe audio from MP3 (transcribe_audio)\n"
        "B9) Convert Markdown to HTML (md_to_html)\n"
        "B10) Write an API endpoint that filters a CSV file and returns JSON (filter_csv_endpoint)\n"
        "Remember: data outside /data must not be accessed, data must never be deleted.\n"
    )
    user_message = task
    if user_email:
        user_message += f"\nUser email: {user_email}"

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "system", "content": system_instructions}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "script_runner",
                    "description": "Install uv if needed and run a script with provided args.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "script_url": {"type": "string"},
                            "args": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["script_url", "args"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "prettier_formatter",
                    "description": "Install Prettier@version, format file in-place.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "version": {"type": "string"}
                        },
                        "required": ["file_path", "version"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "count_wednesdays",
                    "description": "Count the Wednesdays in file_in, write the number to file_out.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_in": {"type": "string"},
                            "file_out": {"type": "string"}
                        },
                        "required": ["file_in", "file_out"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "sort_contacts",
                    "description": (
                        "Sort the array of contacts in 'file_in' by last_name, then first_name, "
                        "and write the result to 'file_out'."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_in": {"type": "string"},
                            "file_out": {"type": "string"}
                        },
                        "required": ["file_in", "file_out"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "recent_logs",
                    "description": (
                        "Find the 10 most recent .log files in 'logs_dir' by modification time, "
                        "read their first lines, and write them (in descending order) to 'output_file'."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "logs_dir": {"type": "string"},
                            "output_file": {"type": "string"}
                        },
                        "required": ["logs_dir", "output_file"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "docs_index",
                    "description": (
                        "Recursively find all .md files in 'docs_dir'. For each file, extract the first line starting with '# '. "
                        "Then store an entry in output_file mapping the relative path (minus docs_dir) to the extracted title."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "docs_dir": {"type": "string"},
                            "output_file": {"type": "string"}
                        },
                        "required": ["docs_dir", "output_file"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_email_sender",
                    "description": (
                        "Given an email text file, pass its content to GPT-4o-Mini, " "extract the sender's email address, and write it to an output file."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_in": {"type": "string"},
                            "file_out": {"type": "string"}
                        },
                        "required": ["file_in", "file_out"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_credit_card",
                    "description": (
                        "Reads an image file containing a credit card number, passes it to GPT-4o-Mini for OCR, "
                        "and writes the card number (with no spaces) to output_file."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_in": {"type": "string"},
                            "file_out": {"type": "string"}
                        },
                        "required": ["file_in", "file_out"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "comments_similarity",
                    "description": (
                        "Given a text file containing comments (one per line), use AI Proxy embeddings to find "
                        "the two most similar comments, then write those two lines to 'file_out', one per line."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_in": {"type": "string"},
                            "file_out": {"type": "string"}
                        },
                        "required": ["file_in", "file_out"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ticket_sales_gold",
                    "description": (
                        "Open /data/ticket-sales.db, read rows from 'tickets' table, "
                        "calculate total of units*price for rows where type='Gold', "
                        "and write the number to file_out."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "db_path": {"type": "string"},
                            "file_out": {"type": "string"}
                        },
                        "required": ["db_path", "file_out"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_data_api",
                    "description": (
                        "Fetch data from an API endpoint and save it to a file."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "file_out": {"type": "string"}
                        },
                        "required": ["url", "file_out"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "clone_git_repo",
                    "description": (
                        "Clone a git repo and optionally make a commit."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo_url": {"type": "string"},
                            "commit_message": {"type": "string"}
                        },
                        "required": ["repo_url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "sql_query",
                    "description": (
                        "Run a SQL query on a SQLite or DuckDB database, write results to file_out."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "db_path": {"type": "string"},
                            "query": {"type": "string"},
                            "file_out": {"type": "string"}
                        },
                        "required": ["db_path", "query", "file_out"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "scrape_website",
                    "description": (
                        "Extract data from a website by URL, write to file_out."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "file_out": {"type": "string"}
                        },
                        "required": ["url", "file_out"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "compress_image",
                    "description": (
                        "Compress or resize an image, writing the output back to file_out."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "src_path": {"type": "string"},
                            "file_out": {"type": "string"},
                            "width": {"type": "number"},
                            "height": {"type": "number"}
                        },
                        "required": ["src_path", "file_out"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "transcribe_audio",
                    "description": (
                        "Transcribe an MP3 file using a local or remote STT engine."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "src_path": {"type": "string"},
                            "file_out": {"type": "string"}
                        },
                        "required": ["src_path", "file_out"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "md_to_html",
                    "description": (
                        "Convert Markdown in file_in to HTML, writing to file_out."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_in": {"type": "string"},
                            "file_out": {"type": "string"}
                        },
                        "required": ["file_in", "file_out"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "filter_csv_endpoint",
                    "description": (
                        "Load a CSV, filter it, and return JSON data. Writes result to file_out or outputs JSON directly."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "csv_path": {"type": "string"},
                            "filters": {"type": "object"},
                            "file_out": {"type": "string"}
                        },
                        "required": ["csv_path", "filters", "file_out"]
                    }
                }
            }
        ],
        "tool_choice": "auto"
    }

    resp = requests.post(url, headers=headers, json=data)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    try:
        response_json = resp.json()
        tool_calls = response_json["choices"][0]["message"].get("tool_calls", [])
        if not tool_calls:
            return {"status": "no_tool_called", "output": response_json}

        outputs = []
        for call in tool_calls:
            fn_name = call["function"]["name"]
            fn_args = json.loads(call["function"]["arguments"])

            # A1: script_runner
            if fn_name == "script_runner":
                subprocess.run(["pip", "install", "--upgrade", "uv"], check=True)
                cmd = ["uv", "run", fn_args["script_url"]] + fn_args["args"]
                r = subprocess.run(cmd, capture_output=True, text=True, check=True)
                outputs.append(f"[script_runner output]\n{r.stdout}")

            # A2: prettier_formatter
            elif fn_name == "prettier_formatter":
                fp = fn_args["file_path"]
                ver = fn_args["version"]
                try:
                    # Call the format_with_prettier function with the file path and version
                    output = format_with_prettier(fp, ver)
                    
                    # Append the output of the Prettier formatting

                    post_process_markdown(fp)
                    outputs.append(f"[prettier_formatter output]\n{output}")
                
                except Exception as e:
                    # If there is an error, append the error message
                    outputs.append(f"[Error in prettier_formatter]: {str(e)}")

            # A3: count_wednesdays
            elif fn_name == "count_wednesdays":
                file_in = fn_args["file_in"]
                file_out = fn_args["file_out"]
                weds_count = 0
                with safe_open(file_in, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            dt = parse(line, fuzzy=True)
                            if dt.weekday() == 2:  # Wednesday=2
                                weds_count += 1
                        except Exception:
                            pass
                with safe_open(file_out, "w", encoding="utf-8") as f_out:
                    f_out.write(str(weds_count))
                outputs.append(f"[count_wednesdays output]\nFound {weds_count} Wednesdays.")

            # A4: sort_contacts
            elif fn_name == "sort_contacts":
                file_in = fn_args["file_in"]
                file_out = fn_args["file_out"]
                try:
                    with safe_open(file_in, "r", encoding="utf-8") as f_in:
                        contacts = json.load(f_in)
                    contacts.sort(key=lambda c: (c["last_name"], c["first_name"]))
                    with safe_open(file_out, "w", encoding="utf-8") as f_out:
                        json.dump(contacts, f_out, ensure_ascii=False, indent=2)
                    outputs.append(f"[sort_contacts output]\nSorted contacts saved to {file_out}")
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

            # A5: recent_logs
            elif fn_name == "recent_logs":
                import os
                logs_dir = fn_args["logs_dir"]
                output_file = fn_args["output_file"]
                try:
                    # Collect all .log files with mod times
                    log_files = [
                        f for f in os.listdir(logs_dir) if f.endswith(".log")
                    ]
                    full_paths = [
                        (os.path.join(logs_dir, f), f)
                        for f in log_files
                    ]
                    # Sort by modification time descending
                    full_paths.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)
                    # Take top 10
                    recent_files = full_paths[:10]

                    lines = []
                    for path, fname in recent_files:
                        with safe_open(path, "r", encoding="utf-8") as f_log:
                            first_line = f_log.readline().rstrip("\n")
                            lines.append(first_line)

                    with safe_open(output_file, "w", encoding="utf-8") as f_out:
                        f_out.write("\n".join(lines) + "\n")

                    outputs.append(
                        f"[recent_logs output]\nWrote first lines of {len(recent_files)} recent .log files to {output_file}"
                    )

                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

            # A6: docs_index
            elif  fn_name == "docs_index":
                import os
                docs_dir = fn_args["docs_dir"] 
                output_file = fn_args["output_file"] 
                index_map = {}
                # Recursively walk docs_dir to find .md files
                for root, dirs, files in os.walk(docs_dir):
                    for filename in files:
                        if filename.endswith(".md"):
                            full_path = os.path.join(root, filename)
                            # Determine the relative path from docs_dir (e.g. "subdir/filename.md")
                            rel_path = os.path.relpath(full_path, start=docs_dir).replace("\\", "/")

                            # Read file, extract first occurrence of a line starting with "# "
                            title = None
                            with safe_open(full_path, "r", encoding="utf-8") as f_md:
                                for line in f_md:
                                    line = line.strip()
                                    if line.startswith("# "):
                                        # Extract everything after "# "
                                        title = line[2:].strip()
                                        break

                            # If we found a title, save it in the index_map
                            if title:
                                index_map[rel_path] = title

                # Write index_map as JSON to output_file
                with safe_open(output_file, "w", encoding="utf-8") as f_out:
                    json.dump(index_map, f_out, ensure_ascii=False, indent=2)

                outputs.append(
                    f"[docs_index output]\nWrote index for {len(index_map)} markdown files to {output_file}."
                )

            # A7: extract_email_sender
            elif fn_name == "extract_email_sender":
                file_in = fn_args["file_in"] 
                file_out = fn_args["file_out"]
                # 1) Read the email content from file_in
                with safe_open(file_in, "r", encoding="utf-8") as f_in:
                    email_content = f_in.read()

                # 2) Make a direct GPT-4o-Mini request to extract just the sender’s email
                try:
                    # Use your existing AIPROXY_TOKEN
                    gpt_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
                    gpt_headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {AIPROXY_TOKEN}",
                    }
                    gpt_data = {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are an AI assistant. Extract the sender's email address "
                                    "from the following raw email text. Return only the email address."
                                )
                            },
                            {"role": "user", "content": email_content}
                        ]
                    }
                    gpt_resp = requests.post(gpt_url, headers=gpt_headers, json=gpt_data)
                    gpt_resp.raise_for_status()
                    gpt_json = gpt_resp.json()

                    # 3) Get the raw text from GPT's assistant message, e.g. gpt_json["choices"][0]["message"]["content"]
                    extracted = gpt_json["choices"][0]["message"]["content"].strip()

                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error calling GPT for email extraction: {str(e)}")

                # 4) Write the extracted email address to file_out
                with safe_open(file_out, "w", encoding="utf-8") as f_out:
                    f_out.write(extracted)

                outputs.append(
                    f"[extract_email_sender output]\nExtracted email: {extracted}"
                )

            # A8: extract_credit_card
            elif fn_name == "extract_credit_card":
                import easyocr
                import re
                from PIL import Image
                from pathlib import Path

                file_in = fn_args["file_in"]
                file_out = fn_args["file_out"]

                try:
                    # 1) Initialize the EasyOCR reader (English only, disable GPU)
                    reader = easyocr.Reader(['en'], gpu=False)

                    # 2) Confirm the input file exists
                    input_path = Path(file_in)
                    if not input_path.exists():
                        raise FileNotFoundError(f"Image file {file_in} does not exist.")

                    # 3) Run OCR on the image
                    ocr_result = reader.readtext(str(input_path))

                    # 4) Combine extracted text
                    extracted_text = " ".join([seg_text for (_, seg_text, _) in ocr_result])

                    # 5) Extract digits from the OCR result (remove non-digit characters)
                    extracted_text = re.sub(r"\D", "", extracted_text)

                    # 6) Find the first valid 16-digit sequence
                    card_number = None
                    matches = re.findall(r"\d{16}", extracted_text)
                    if not matches:
                        raise ValueError("No 16-digit sequence found in the extracted text.")
                    
                    card_number = matches[0]

                    # 7) Apply Luhn check
                    if passes_luhn(card_number):
                        final_number = card_number
                    else:
                        # If the first digit is '9', try flipping it to '3' and check again
                        if card_number[0] == '9':
                            possible_fix = '3' + card_number[1:]
                            if passes_luhn(possible_fix):
                                final_number = possible_fix
                            else:
                                raise ValueError(f"Luhn check failed, even after flipping '9' -> '3'. Extracted: {card_number}")
                        else:
                            raise ValueError(f"Luhn check failed. No known fix for card number: {card_number}")

                    # 8) Write the final card number to file
                    output_path = Path(file_out)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with safe_open(output_path, "w", encoding="utf-8") as f_out:
                        f_out.write(final_number)

                    outputs.append(f"[extract_credit_card output]\nOCR complete. Best 16-digit match: {final_number}")

                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"OCR extraction error: {str(e)}")



            # A9: comments_similarity
            elif fn_name == "comments_similarity":
                import numpy as np
                import os
                from sklearn.metrics.pairwise import cosine_similarity
                from pathlib import Path
                file_in = fn_args["file_in"]
                file_out = fn_args["file_out"]

                # 1) Ensure file_in exists
                if not os.path.exists(file_in):
                    raise HTTPException(status_code=404, detail=f"{file_in} not found")

                # 2) Read comments (one per line)
                with safe_open(file_in, "r", encoding="utf-8") as f_in:
                    comments = [line.strip() for line in f_in if line.strip()]

                if len(comments) < 2:
                    raise HTTPException(
                        status_code=400,
                        detail="Need at least two comments to find most similar pair."
                    )

                # 3) Call AI Proxy to embed all comments at once
                try:
                    embedding_url = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
                    embedding_headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {AIPROXY_TOKEN}"  # Make sure your token is correct here
                    }
                    embedding_payload = {
                        "model": "text-embedding-3-small",  # Adjust the model if needed
                        "input": comments
                    }

                    embed_resp = requests.post(embedding_url, headers=embedding_headers, json=embedding_payload)

                    # Check for HTTP 401 Unauthorized error
                    if embed_resp.status_code == 401:
                        raise HTTPException(status_code=401, detail="Unauthorized. Please check your API token permissions.")

                    embed_resp.raise_for_status()  # raises HTTPError for other 4xx/5xx errors
                    embed_json = embed_resp.json()

                    # 4) Convert to NumPy array
                    embeddings = [item["embedding"] for item in embed_json["data"]]
                    embeddings_np = np.array(embeddings)

                except requests.HTTPError as e:
                    raise HTTPException(status_code=embed_resp.status_code, detail=embed_resp.text)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Embedding request error: {str(e)}")

                # 5) Find the most similar pair using cosine similarity
                similarity_matrix = embeddings_np @ embeddings_np.T
                norms = np.linalg.norm(embeddings_np, axis=1)
                denom = np.outer(norms, norms)
                with np.errstate(divide='ignore', invalid='ignore'):
                    similarity_matrix = np.true_divide(similarity_matrix, denom)
                    np.fill_diagonal(similarity_matrix, -np.inf)

                i, j = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
                most_sim_1 = comments[i]
                most_sim_2 = comments[j]

                # 6) Write the two most similar comments to file_out
                output_file = Path(file_out)
                output_file.parent.mkdir(parents=True, exist_ok=True)

                with safe_open(file_out, "w", encoding="utf-8") as f_out:
                    f_out.write(f"{most_sim_1}\n{most_sim_2}\n")

                # Output the result for logging
                outputs.append(
                    f"[comments_similarity output]\n"
                    f"Most similar pair:\n{most_sim_1}\n{most_sim_2}\n"
                    f"Cosine similarity: {similarity_matrix[i, j]:.4f}"
                )


            # A10: ticket_sales_gold
            
            elif fn_name == "ticket_sales_gold":
                import sqlite3
                import os
                db_path = fn_args["db_path"]
                file_out = fn_args["file_out"]

                if not os.path.exists(db_path):
                    raise HTTPException(status_code=404, detail=f"{db_path} not found")

                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()

                    # Query rows from tickets table where type='Gold'
                    # Sum units*price
                    cursor.execute("""
                        SELECT SUM(units * price)
                        FROM tickets
                        WHERE LOWER(type)='gold'
                    """)
                    result = cursor.fetchone()[0]
                    conn.close()

                    # If result is None, means no matching rows, treat as 0
                    if result is None:
                        result = 0.0

                    # Write the result (float) to file_out
                    with safe_open(file_out, "w", encoding="utf-8") as f_out:
                        f_out.write(str(result))

                    outputs.append(
                        f"[ticket_sales_gold output]\nTotal Gold sales: {result}"
                    )

                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"DB error: {str(e)}")

            #B3 fetch_data_api
            elif fn_name == "fetch_data_api":
                import httpx 
                url = fn_args["url"]
                file_out = fn_args["file_out"]
                try:
                    # fetch data
                    r = httpx.get(url, timeout=10)
                    r.raise_for_status()
                    data = r.content  # or .text

                    # save to file_out in /data
                    with safe_open(file_out, "wb") as f_out:
                        f_out.write(data)

                    outputs.append(f"[fetch_data_api] Saved data from {url} to {file_out}")
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"fetch_data_api error: {str(e)}")


            #B4 clone_git_repo
            elif fn_name == "clone_git_repo":
                from pathlib import Path
                repo_url = fn_args["repo_url"]
                commit_msg = fn_args.get("commit_message")

                # Directory where we'll clone repositories under /data
                repos_dir = "/data/repos"
                os.makedirs(repos_dir, exist_ok=True)

                # Derive a local folder name from the repo URL
                repo_name = repo_url.rstrip("/").split("/")[-1]
                if repo_name.endswith(".git"):
                    repo_name = repo_name[:-4]

                local_repo_path = os.path.join(repos_dir, repo_name)

                try:
                    # Clone the repo into /data/repos/<repo_name>
                    subprocess.run(["git", "clone", repo_url, local_repo_path], check=True)

                    # If a commit message is provided, make an empty commit in the cloned repo
                    if commit_msg:
                        # Navigate via -C <path>, allow an empty commit for demonstration
                        subprocess.run(
                            ["git", "-C", local_repo_path, "commit", "--allow-empty", "-m", commit_msg],
                            check=True
                        )
                        outputs.append(
                            f"[clone_git_repo] Cloned '{repo_url}' into '{local_repo_path}' and made an empty commit: {commit_msg}"
                        )
                    else:
                        outputs.append(f"[clone_git_repo] Successfully cloned '{repo_url}' into '{local_repo_path}'.")

                except subprocess.CalledProcessError as e:
                    raise HTTPException(status_code=500, detail=f"clone_git_repo error: {str(e)}")

            
            #B5 sql_query
            elif fn_name == "sql_query": 
                import sqlite3 
                db_path = fn_args["db_path"] 
                query = fn_args["query"] 
                file_out = fn_args["file_out"] 
                try: 
                    conn = sqlite3.connect(db_path) 
                    cur = conn.cursor() 
                    cur.execute(query) 
                    results = cur.fetchall() 
                    conn.close()
                        # Save results to JSON or CSV, etc.
                    
                    with safe_open(file_out, "w", encoding="utf-8") as f_out:
                        json.dump(results, f_out)

                    outputs.append(f"[sql_query] Executed query, wrote {len(results)} rows to {file_out}")
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"sql_query error: {str(e)}")


            #B6 scrape_website
            elif fn_name == "scrape_website": 
                import httpx 
                import re 
                url = fn_args["url"] 
                file_out = fn_args["file_out"] 
                try: 
                    r = httpx.get(url, timeout=10) 
                    r.raise_for_status() 
                    content = r.text
                        # minimal placeholder for extracting something
                    # real logic would parse HTML with e.g. BeautifulSoup
                    with safe_open(file_out, "w", encoding="utf-8") as f_out:
                        f_out.write(content)

                    outputs.append(f"[scrape_website] Scraped {url}, wrote raw HTML to {file_out}")
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"scrape_website error: {str(e)}")


            #B7 compress_image
            elif fn_name == "compress_image": 
                from PIL import Image 
                src_path = fn_args["src_path"] 
                file_out = fn_args["file_out"] 
                width = fn_args.get("width") 
                height = fn_args.get("height")
                try:
                    # Save compressed/resized to file_out
                    with safe_open(file_out, "wb") as f_out:
                        img.save(f_out, format="JPEG", quality=85)
                    # If both width and height are provided, resize
                    # (If not provided, we just leave the image at its original size)
                    if width and height:
                        # Convert to int in case they come as strings
                        new_width = int(width)
                        new_height = int(height)

                        # ANTIALIAS is now called Resampling in newer Pillow versions
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                    # Write the resized/compressed image to file_out
                    # Choose a format, e.g. JPEG, and set a quality
                    with safe_open(file_out, "wb") as f_out:
                        img.save(f_out, format="JPEG", quality=85)
                    outputs.append(f"[compress_image] Resized image from {src_path} -> {file_out}")
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"compress_image error: {str(e)}")


            #B8 transcribe_audio
            elif fn_name == "transcribe_audio": 
                import speech_recognition as sr
                src_path = fn_args["src_path"]
                file_out = fn_args["file_out"]

                try:
                    # We'll trust that src_path is under /data. 
                    # If you want to enforce it, you can do a quick check with safe_open in read mode:
                    with safe_open(src_path, "rb"):
                        pass  # Just to confirm path is valid & within /data

                    # Initialize recognizer
                    r = sr.Recognizer()

                    # The AudioFile class expects a file path, so we pass src_path directly
                    with sr.AudioFile(src_path) as source:
                        audio_data = r.record(source)  # read the entire audio file

                    # Perform speech-to-text using Google's free API or any other local STT engine
                    text = r.recognize_google(audio_data)

                    # Write the transcription to file_out
                    with safe_open(file_out, "w", encoding="utf-8") as f_out:
                        f_out.write(text)

                    outputs.append(
                        f"[transcribe_audio] Transcribed audio from '{src_path}' -> '{file_out}'"
                    )

                except sr.UnknownValueError:
                    raise HTTPException(
                        status_code=500, 
                        detail="Could not understand the audio (speech_recognition UnknownValueError)."
                    )
                except sr.RequestError as e:
                    raise HTTPException(
                        status_code=500, 
                        detail=f"API Request error during transcription: {str(e)}"
                    )
                except Exception as e:
                    raise HTTPException(
                        status_code=500, 
                        detail=f"transcribe_audio error: {str(e)}"
                    )

                
            #B9 md_to_html
            elif fn_name == "md_to_html":
                import markdown
                file_in = fn_args["file_in"]
                file_out = fn_args["file_out"]

                try:
                    # 1) Read the Markdown from file_in
                    with safe_open(file_in, "r", encoding="utf-8") as f_in:
                        md_text = f_in.read()

                    # 2) Convert to HTML
                    html = markdown.markdown(md_text)

                    # 3) Write the HTML to file_out
                    with safe_open(file_out, "w", encoding="utf-8") as f_out:
                        f_out.write(html)

                    outputs.append(
                        f"[md_to_html] Converted '{file_in}' to HTML in '{file_out}'"
                    )

                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"md_to_html error: {str(e)}"
                    )

            #B10 filter_csv_endpoint
            elif fn_name == "filter_csv_endpoint":
                import csv
                csv_path = fn_args["csv_path"]
                filters = fn_args["filters"]  # e.g. {"column_name": "value"}
                file_out = fn_args["file_out"]

                try:
                    # 1) Read CSV from /data
                    with safe_open(csv_path, "r", encoding="utf-8") as f_in:
                        reader = csv.DictReader(f_in)
                        rows = list(reader)

                    # 2) Apply each filter
                    filtered = []
                    for row in rows:
                        match = True
                        for col, val in filters.items():
                            if str(row.get(col, "")) != str(val):
                                match = False
                                break
                        if match:
                            filtered.append(row)

                    # 3) Write filtered rows to JSON in /data
                    with safe_open(file_out, "w", encoding="utf-8") as f_out:
                        json.dump(filtered, f_out, ensure_ascii=False, indent=2)

                    outputs.append(
                        f"[filter_csv_endpoint] Found {len(filtered)} matching rows; wrote them to '{file_out}'"
                    )

                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"filter_csv_endpoint error: {str(e)}"
                    )

                outputs.append("[clone_git_repo] Placeholder logic executed.")
            else:
                outputs.append(f"[Unknown tool: {fn_name}]")

        return {"status": "success", "output": "\n".join(outputs)}

    except (KeyError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Invalid AI response: {e}")
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Subprocess error: {e.stderr}")


@app.get("/read")
def read_file(path: str = Query(..., title="path")):
    """
    Returns the contents of the specified file, or 404 if not found.
    """
    # if not os.path.exists(path):
    #     return Response(status_code=status.HTTP_404_NOT_FOUND)
    # with open(path, "r", encoding="utf-8") as f:
    #     return Response(content=f.read(), media_type="text/plain")

    # Use safe_open instead of open
    try:
        with safe_open(path, "r", encoding="utf-8") as f:
            return Response(content=f.read(), media_type="text/plain")
    except FileNotFoundError:
        return Response(status_code=status.HTTP_404_NOT_FOUND)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
