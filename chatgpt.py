from prompt import qa_template
from model import llm
from data import documents
import gradio as gr

qa_template = qa_template.partial(context=documents)
chat_history = []

def chat(query):
    full_prompt = qa_template.format(question=query)
    print(full_prompt)
    return llm.invoke(full_prompt)

gr.Interface(fn=chat, inputs="text", outputs="text").launch()