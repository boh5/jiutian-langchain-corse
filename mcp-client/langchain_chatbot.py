import asyncio
import json
import logging
import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient


class Configuration:
    """读取 .env 与 servers_config.json"""

    def __init__(self) -> None:
        load_dotenv()
        self.api_key: str = os.environ["LLM_API_KEY"]
        self.base_url: str | None = os.environ["BASE_URL"]
        self.model: str = os.environ["MODEL"]
        if not self.api_key:
            raise ValueError("❌ 未找到 LLM_API_KEY，请在 .env 中配置")

    @staticmethod
    def load_servers(file_path: str = "servers_config.json") -> Dict[str, Any]:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f).get("mcpServers", {})


async def run_chat_loop() -> None:
    cfg = Configuration()

    servers_cfg = Configuration.load_servers()

    mcp_client = MultiServerMCPClient(servers_cfg)

    tools = await mcp_client.get_tools()

    logging.info(f"✅ 已加载 {len(tools)} 个 MCP 工具: {[t.name for t in tools]}")

    llm = init_chat_model(
        model=cfg.model,
        model_provider="openai",
        base_url=cfg.base_url,
        api_key=cfg.api_key,
    )

    prompt = hub.pull("hwchase17/openai-tools-agent")
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    print("\n🤖 MCP Agent 已启动，输入 'quit' 退出")
    while True:
        user_input = input("\n你：").strip()
        if user_input.lower() == "quit":
            break
        try:
            result = await agent_executor.ainvoke({"input": user_input})
            print(f"\nAI: {result['output']}")
        except Exception as e:
            print(f"\n⚠ 出错: {e}")

    print("🧹 资源已清理，Bye!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    asyncio.run(run_chat_loop())
