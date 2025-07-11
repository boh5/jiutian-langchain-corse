import gradio as gr
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

model = init_chat_model(
    model="deepseek-v3",
    model_provider="openai",
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

system_prompt = ChatPromptTemplate.from_messages([
    ("system", "你叫小智，是一名乐于助人的助手。"),
    ("human", "{input}"),
])

qa_chain = system_prompt | model | StrOutputParser()

async def chat_response(message, history):
    partial_message = ""

    async for chunk in qa_chain.astream({"input": message}):
        partial_message += chunk
        yield partial_message

# ──────────────────────────────────────────────
# 2. Gradio 组件
# ──────────────────────────────────────────────
CSS = """
.main-container {max-width: 1200px; margin: 0 auto; padding: 20px;}
.header-text {text-align: center; margin-bottom: 20px;}
"""

def create_chatbot() -> gr.Blocks:
    with gr.Blocks(title="DeepSeek Chat", css=CSS) as demo:
        with gr.Column(elem_classes=["main-container"]):
            gr.Markdown("# 🤖 LangChain B站公开课 By九天Hector", elem_classes=["header-text"])
            gr.Markdown("基于 LangChain LCEL 构建的流式对话机器人", elem_classes=["header-text"])

            chatbot = gr.Chatbot(
                height=500,
                show_copy_button=True,
                avatar_images=(
                    "https://cdn.jsdelivr.net/gh/twitter/twemoji@v14.0.2/assets/72x72/1f464.png",
                    "https://cdn.jsdelivr.net/gh/twitter/twemoji@v14.0.2/assets/72x72/1f916.png",
                ),
            )
            msg = gr.Textbox(placeholder="请输入您的问题...", container=False, scale=7)
            submit = gr.Button("发送", scale=1, variant="primary")
            clear = gr.Button("清空", scale=1)

        # ---------------  状态：保存 messages_list  ---------------
        state = gr.State([])          # 这里存放真正的 Message 对象列表

        # ---------------  主响应函数（流式） ----------------------
        async def respond(user_msg: str, chat_hist: list, messages_list: list):
            # 1) 输入为空直接返回
            if not user_msg.strip():
                yield "", chat_hist, messages_list
                return

            # 2) 追加用户消息
            messages_list.append(HumanMessage(content=user_msg))
            chat_hist = chat_hist + [(user_msg, None)]
            yield "", chat_hist, messages_list      # 先显示用户消息

            # 3) 流式调用模型
            partial = ""
            async for chunk in qa_chain.astream({"messages": messages_list}):
                partial += chunk
                # 更新最后一条 AI 回复
                chat_hist[-1] = (user_msg, partial)
                yield "", chat_hist, messages_list

            # 4) 完整回复加入历史，裁剪到最近 50 条
            messages_list.append(AIMessage(content=partial))
            messages_list = messages_list[-50:]

            # 5) 最终返回（Gradio 需要把新的 state 传回）
            yield "", chat_hist, messages_list

        # ---------------  清空函数 -------------------------------
        def clear_history():
            return [], "", []          # 清空 Chatbot、输入框、messages_list

        # ---------------  事件绑定 ------------------------------
        msg.submit(respond, [msg, chatbot, state], [msg, chatbot, state])
        submit.click(respond, [msg, chatbot, state], [msg, chatbot, state])
        clear.click(clear_history, outputs=[chatbot, msg, state])

    return demo


# ──────────────────────────────────────────────
# 3. 启动应用
# ──────────────────────────────────────────────
demo = create_chatbot()
demo.launch(server_name="localhost", server_port=7860, share=False, debug=True)
