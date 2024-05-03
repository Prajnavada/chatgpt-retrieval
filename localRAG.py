from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

from model import llm
from prompt import qa_prompt
from data import documents

import gradio as gr

embeddings = GPT4AllEmbeddings() # open source, fast
# embeddings = SentenceTransformerEmbeddings()

db = FAISS.from_documents(documents, embeddings)

def chat(query):
    result = db.similarity_search(query)
    context = result[0].page_content
    full_prompt = qa_prompt.format(context=context, question=query)
    response = llm.invoke(full_prompt)
    return response

gr.Interface(fn=chat, inputs="text", outputs="text").launch()

# query = "Do I like vegetables?"
# docs = db.similarity_search(query)
# print(docs[0].page_content)

# while True:
#     question = input("You: ")
#     response = chat(question)
#     print("Bot: " + response)