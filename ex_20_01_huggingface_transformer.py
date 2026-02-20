# Use a pipeline as a high-level helper
import os
from dotenv import load_dotenv
from transformers import pipeline


# Load .env FIRST
load_dotenv()

pipe = pipeline("text-generation", model="ByteDance-Seed/Stable-DiffCoder-8B-Instruct", trust_remote_code=True, token=os.getenv("HF_TOKEN"))
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)