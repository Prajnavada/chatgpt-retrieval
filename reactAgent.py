# import gradio as gr

from model import llm
from prompt import react_prompt, react_chat_prompt, react_chat_json_prompt, react_multi_input_prompt, structured_chat_prompt

from langchain.agents import AgentExecutor, create_react_agent, create_structured_chat_agent, create_tool_calling_agent, create_json_chat_agent

### tools
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser
sync_browser = create_sync_playwright_browser(headless=False)
toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
tools = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser).get_tools()

# agent = create_react_agent(llm, tools, react_prompt)
# agent = create_json_chat_agent(llm, tools, react_chat_json_prompt)
agent = create_structured_chat_agent(llm, tools, structured_chat_prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def chat(query):
    result = agent_executor.invoke({
        "input":query,
        "chat_history": []
    })
    print(result)
    return result

# gr.Interface(fn=chat, inputs="text", outputs="text").launch()

while True:
    question = input("You: ")
    response = chat(question)
    print(response)