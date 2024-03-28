from langchain_community.document_loaders import DirectoryLoader
from langchain_openai.llms import OpenAI

import constants
import gradio as gr

llm = OpenAI(openai_api_key=constants.APIKEY)

from langchain.prompts import PromptTemplate
qa_doc = """
Based on the following context, answer the question below.

**Context:**

{context}

**Question:**

{question}

**Answer:**
"""
qa_template = PromptTemplate(template=qa_doc,
                             input_variables=["context", "question"])

# Method to load data from the data folder as text
def load_data():
    loader = DirectoryLoader("data")
    content = ""
    for doc in loader.load():
        content += doc.page_content + "\n\n"
    return content

qa_template = qa_template.partial(context=load_data())
chat_history = []

def chat(query):
    # user_input = input("User: ")
    full_prompt = qa_template.format(question=query)
    print(full_prompt)
    return llm.invoke(full_prompt)

# Either this or the gr.Interface.
# while True:
#     question = input("You: ")
#     response = chat(question)
#     print("Bot: " + response)

gr.Interface(fn=chat, inputs="text", outputs="text").launch()