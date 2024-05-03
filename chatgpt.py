from model import llm
from prompt import qa_prompt
from data import documents
import gradio as gr

qa_template = qa_prompt.partial(context=documents)
chat_history = []

def chat(query):
    full_prompt = qa_template.format(question=query)
    # print(full_prompt)
    return llm.invoke(full_prompt)

gr.Interface(fn=chat, inputs="text", outputs="text").launch()