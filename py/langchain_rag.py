import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.agents import (
    AgentExecutor,
    create_tool_calling_agent,
)
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
import streamlit as st

load_dotenv()

DASHSCOPE_API_KEY = os.environ["DASHSCOPE_API_KEY"]

# 设置环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

embeddings = DashScopeEmbeddings(
    model="text-embedding-v1", dashscope_api_key=DASHSCOPE_API_KEY
)


def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")


def get_conversational_chain(tool, ques):
    llm = init_chat_model(
        model="deepseek-v3",
        model_provider="openai",
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                '你是AI助手，请根据提供的上下文回答问题，确保提供所有细节，如果答案不在上下文中，请说、"答案不在上下文中"，不要提供错误的答案',
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    tools = [tool]
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response = agent_executor.invoke({"input": ques})
    print(response)
    st.write("🤖回答: ", response["output"])


def check_database_exists():
    return os.path.exists("faiss_db") and os.path.exists("faiss_db/index.faiss")


def user_input(user_question):
    if not check_database_exists():
        st.error("❌ 请先上传PDF文件并点击'Submit & Process'按钮来处理文档！")
        st.info("💡 步骤：1️⃣ 上传PDF → 2️⃣ 点击处理 → 3️⃣ 开始提问")
        return

    try:
        new_db = FAISS.load_local(
            "faiss_db", embeddings, allow_dangerous_deserialization=True
        )

        retriever = new_db.as_retriever()
        retrieval_chain = create_retriever_tool(
            retriever,
            "pdf_extractor",
            "This tool is to give answer to queries from the pdf",
        )
        get_conversational_chain(retrieval_chain, user_question)
    except Exception as e:
        st.error(f"❌ 加载数据库时出错: {str(e)}")
        st.info("请重新处理PDF文件")


def main():
    st.set_page_config("🤖 LangChain RAG")
    st.header("🤖 LANGCHAIN RAG BOT")

    # 显示数据库状态
    col1, col2 = st.columns([3, 1])

    with col1:
        if not check_database_exists():
            st.warning("⚠ 请先上传并处理 PDF 文件")

    with col2:
        if st.button("🗑 清除数据库"):
            try:
                import shutil

                if os.path.exists("faiss_db"):
                    shutil.rmtree("faiss_db")
                st.success("数据库已清除")
                st.rerun()
            except Exception as e:
                st.error(f"清除失败: {e}")

    # 用户输入问题
    user_question = st.text_input(
        "💬 请输入问题",
        placeholder="例如：这个文档的主要内容是什么？",
        disabled=not check_database_exists(),
    )

    if user_question:
        if check_database_exists():
            with st.spinner("🤔 AI 正在分析文档..."):
                user_input(user_question)
        else:
            st.error("❌ 请先上传并处理 PDF 文件!")

    # 侧边栏
    with st.sidebar:
        st.title("📁 文档管理")

        if check_database_exists():
            st.success("✅ 数据库状态：已就绪")
        else:
            st.info("📝 状态：等待上传 PDF")

        st.markdown("---")

        # 文件上传
        pdf_doc = st.file_uploader(
            "📎上传 PDF 文件",
            accept_multiple_files=True,
            type=["pdf"],
            help="支持上传多个PDF文件",
        )
        if pdf_doc:
            st.info(f"📄 已选择 {len(pdf_doc)} 个文件")
            for i, pdf in enumerate(pdf_doc, 1):
                st.write(f"{i}. {pdf.name}")

        # 处理按钮
        process_button = st.button(
            "🚀 提交并处理",
            disabled=not pdf_doc,
            use_container_width=True,
        )
        if process_button:
            if pdf_doc:
                with st.spinner("📊 正在处理 PDF 文件..."):
                    try:
                        raw_text = pdf_read(pdf_doc)

                        if not raw_text.strip():
                            st.error("❌ 无法从 PDF 中提取文本，请检查文件是否有效")
                            return
                        text_chunks = get_chunks(raw_text)
                        st.info(f"📝 文本已分隔为 {len(text_chunks)} 个片段")

                        vector_store(text_chunks)

                        st.success("✅ PDF 处理完成！现在可以开始提问了")
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 处理 PDF 时出错：{str(e)}")
            else:
                st.warning("⚠ 请先选择 PDF 文件")
        # 使用说明
        with st.expander("💡 使用说明"):
            st.markdown("""
            **步骤：**
            1. 📎 上传一个或多个PDF文件
            2. 🚀 点击"Submit & Process"处理文档
            3. 💬 在主页面输入您的问题
            4. 🤖 AI将基于PDF内容回答问题

            **提示：**
            - 支持多个PDF文件同时上传
            - 处理大文件可能需要一些时间
            - 可以随时清除数据库重新开始
            """)


if __name__ == "__main__":
    main()
