from model import llm

from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser
sync_browser = create_sync_playwright_browser(headless=False)
toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
tools = toolkit.get_tools()
tool_args = {tool.name: tool.args for tool in tools}
page = sync_browser.new_page()


from typing import Any, Dict, List

from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, Json
from langchain_core.output_parsers import PydanticOutputParser

class Action(BaseModel):
    tool_name: str = Field(title="Tool Name", description="The tool name to invoke the call")
    tool_input: Dict[str, Any] = Field(title="Tool Input", description="The input schema of the tool input")

class AgentResponse(BaseModel):
    actions: List[Action] = Field(title="Actions", description="Sequential list of action items")
    agent_scratchpad: List[str] = Field(title="Agent scratchpad", description="The scratchpad to store the agent state and actions")

agent_parser = PydanticOutputParser(pydantic_object=AgentResponse)

pydantic_template = """
You are a navigation agent who is replicating a human in navigation. So you replicate the navigating skills similarly where you only remember the top trusted website but have no recollection of the exact page url.
So you always start with navigating to google.com and using navigation interactions to complete the required task.
For actions, you always use on screen content.

Query : {input}

Tool names and input args : {tools}

Agent scratchpad : {agent_scratchpad}

Current Page : {page_content}

Strictly follow the url formats(complete URL), tool input structure and the below response format.

Response format : {response_format}
"""
navigation_prompt = PromptTemplate(template=pydantic_template,
                                       input_variables=["input"],
                                       partial_variables={"agent_scratchpad": [],
                                                          "page_content": "",
                                                          "response_format": agent_parser.get_format_instructions(),
                                                          "tools": tool_args})

# chain = navigation_prompt | llm | agent_parser

def run_action(response, agent_scratchpad):
    for action in response.actions:
        tool = next((t for t in tools if t.name == action.tool_name), None)
        if not tool:
            print("Tool missing" + action.tool_name)
            return
        try:
            parsed_input = tool.get_input_schema().parse_obj(action.tool_input)
            result = tool.invoke(parsed_input.dict())
        except:
            break
        agent_scratchpad.insert(0, result)
        print(result)
    return agent_scratchpad

def start_conversation():
    agent_scratchpad = []
    while True:
        prompt = navigation_prompt.partial(agent_scratchpad=agent_scratchpad, page_content=str(page.content()))
        user_message = input("User: ")
        chain = prompt | llm | agent_parser
        response = chain.invoke(user_message)
        print(response)
        agent_scratchpad = run_action(response, agent_scratchpad)
        print("-" * 100)

if __name__ == "__main__":
    start_conversation()