import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.tools.playwright.utils import create_sync_playwright_browser
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor

load_dotenv(override=True)
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

model = init_chat_model(
    model="deepseek-v3",
    model_provider="openai",
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

sync_browser = create_sync_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
tools = toolkit.get_tools()

prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_openai_tools_agent(model, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    # 定义任务
    command = {
        "input": "访问这个网站 https://github.com/fufankeji/MateGen/blob/main/README_zh.md 并帮我总结一下这个网站的内容"
    }

    # 执行任务
    response = agent_executor.invoke(command)
    print(response)
