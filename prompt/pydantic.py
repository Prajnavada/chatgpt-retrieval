from typing import List, Any

from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, Json
from langchain_core.output_parsers import PydanticOutputParser

class NavigationPathResponse(BaseModel):
    action: str = Field(title="Action", description="One of 'FINAL_ANSWER', 'ACTION'")
    tool: str = Field(title="Tool Name", description="The tool name that will be needed to call")
    tool_input: Json[Any] = Field(title="Tool input", description="The input for the tool in JSON format")

output_parser = PydanticOutputParser(pydantic_object=NavigationPathResponse)

setup_template = """
You are an agent with the access to tools. Given a user query and the current screen contents, you decide on the next set of actions.

Tools:{tools}

Thought: Do I need to use a tool? Yes

Current Page: {current_page}

Response format : {response_format}
"""
self_discovery_select = PromptTemplate(template=setup_template,
                                       input_variables=["task_description", "current_page"],
                                       partial_variables={"response_format": output_parser.get_format_instructions()})