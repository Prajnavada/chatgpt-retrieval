# import gradio as gr

from model import llm
from prompt import self_discovery_adapt, self_discovery_reasoning, self_discovery_select, self_discovery_structure

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

select_chain = self_discovery_select | llm | StrOutputParser()
adapt_chain = self_discovery_adapt | llm | StrOutputParser()
structure_chain = self_discovery_structure | llm | StrOutputParser()
reasoning_chain = self_discovery_reasoning | llm | StrOutputParser()

prompt_maker_chain = (
    RunnablePassthrough.assign(selected_modules=select_chain)
    .assign(adapted_modules=adapt_chain)
    .assign(reasoning_structure=structure_chain)
    .assign(answer=reasoning_chain)
)

def chat(query):
    result = prompt_maker_chain.invoke({
        "task_description": query,
    })
    print(result)
    return result

# gr.Interface(fn=chat, inputs="text", outputs="text").launch()

while True:
    question = input("You: ")
    response = chat(question)
    print(response['answer'])