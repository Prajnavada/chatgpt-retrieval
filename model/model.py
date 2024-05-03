import os

import constants

os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = constants.APIKEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# OpenAI
# from langchain_openai.llms import OpenAI
# llm = OpenAI(openai_api_key=constants.APIKEY)

# Google
# from langchain_google_genai.llms import GoogleGenerativeAI
# os.environ["GOOGLE_API_KEY"] = constants.GOOGLE_API_KEY
# llm = GoogleGenerativeAI(model="gemini-pro")

from langchain_community.llms import Ollama

# llm = Ollama(model="phi3:3.8b-mini-instruct-4k-fp16")
llm = Ollama(model="llama3:8b-instruct-q6_K")
# llm = Ollama(model="llama3:8b-instruct-q8_0", stop=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|reserved_special_token", "assistant"])
# llm = Ollama(model="llama3:8b-instruct-fp16")

# from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
# from transformers import pipeline
#
# pipe = pipeline("text-generation",
#                 # model="cognitivecomputations/dolphin-2_6-phi-2",
#                 # model="Qwen/Qwen1.5-4B-Chat",
#                 model="microsoft/Phi-3-mini-128k-instruct",
#                 trust_remote_code=True,
#                 use_fast=True,
#                 device=0,
#                 # max_new_tokens=512
#                 )
# llm = HuggingFacePipeline(pipeline=pipe)
