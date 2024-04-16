import os

import constants

# OpenAI
# from langchain_openai.llms import OpenAI
# llm = OpenAI(openai_api_key=constants.APIKEY)

# Google
from langchain_google_genai.llms import GoogleGenerativeAI
os.environ["GOOGLE_API_KEY"] = constants.GOOGLE_API_KEY
llm = GoogleGenerativeAI(model="gemini-pro")