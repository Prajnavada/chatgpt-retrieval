from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS

from model import llm
from prompt import qa_template
from data import documents

import gradio as gr

embeddings = GPT4AllEmbeddings()

db = FAISS.from_documents(documents, embeddings)

chat_history = []

def chat(query):
    context = db.similarity_search(query)
    full_prompt = qa_template.format(context=context, question=query)
    return llm.invoke(full_prompt)

gr.Interface(fn=chat, inputs="text", outputs="text").launch()

# Local test for similar document
# query = "Do I like vegetables?"
# docs = db.similarity_search(query)
# print(docs[0].page_content)

# Terminal way for simple startup.
# while True:
#     question = input("You: ")
#     response = chat(question)
#     print("Bot: " + response)