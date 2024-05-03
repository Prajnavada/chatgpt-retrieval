### tools
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser
sync_browser = create_sync_playwright_browser(headless=False)
toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
tools = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser).get_tools()

tools_by_name = {tool.name: tool for tool in tools}