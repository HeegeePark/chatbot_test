import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
import tempfile
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool

# .env 파일 로드
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ✅ SerpAPI 검색 툴 정의
def search_web():
    search = SerpAPIWrapper()

    def run_with_source(query: str) -> str:
        results = search.results(query)
        organic = results.get("organic_results", [])
        formatted = []
        for r in organic[:5]:
            title = r.get("title")
            link = r.get("link")
            source = r.get("source")
            snippet = r.get("snippet")
            if link:
                formatted.append(f"- [{title}]({link}) ({source})\n  {snippet}")
            else:
                formatted.append(f"- {title} (출처: {source})\n  {snippet}")
        return "\n".join(formatted) if formatted else "검색 결과가 없습니다."

    return Tool(
        name="web_search",
        func=run_with_source,
        description="실시간 뉴스 및 웹 정보를 검색할 때 사용합니다. 결과는 제목+출처+링크+간단요약(snippet) 형태로 반환됩니다."
    )

# ✅ PDF 업로드 → 벡터DB → 검색 툴 생성
def load_pdf_files(uploaded_files):
    all_documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        all_documents.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_documents)

    vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    retriever = vector.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        name="pdf_search",
        description="Use this tool to search information from the pdf document"
    )
    return retriever_tool

# ✅ Agent 대화 실행
def chat_with_agent(user_input, agent_executor):
    result = agent_executor({"input": user_input})
    return result['output']

# ✅ 세션별 히스토리 관리
def get_session_history(session_ids):
    if session_ids not in st.session_state.session_history:
        st.session_state.session_history[session_ids] = ChatMessageHistory()
    return st.session_state.session_history[session_ids]

# ✅ 이전 메시지 출력
def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg['role']).write(msg['content'])

# ✅ 메인 실행
def main():
    st.set_page_config(page_title="AI 비서", layout="wide", page_icon="🤖")

    with st.container():
        st.image('./chatbot_logo_hamster.png', use_container_width=True)
        st.markdown('---')
        st.title("안녕하세찌! 🐹✨ RAG를 활용한 'AI 비서 햄톡이' 라고 하찌! 🐹")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "session_history" not in st.session_state:
        st.session_state["session_history"] = {}

    with st.sidebar:
        st.session_state["OPENAI_API"] = st.text_input(
            "OPENAI API 키", placeholder="Enter Your API Key", type="password"
        )
        st.session_state["SERPAPI_API"] = st.text_input(
            "SERPAPI_API 키", placeholder="Enter Your API Key", type="password"
        )
        st.markdown('---')
        pdf_docs = st.file_uploader(
            "Upload your PDF Files", accept_multiple_files=True, key="pdf_uploader"
        )

    # ✅ 키 입력 확인
    if st.session_state["OPENAI_API"] and st.session_state["SERPAPI_API"]:
        os.environ['OPENAI_API_KEY'] = st.session_state["OPENAI_API"]
        os.environ['SERPAPI_API_KEY'] = st.session_state["SERPAPI_API"]

        # 도구 정의
        tools = []
        if pdf_docs:
            pdf_search = load_pdf_files(pdf_docs)
            tools.append(pdf_search)
        tools.append(search_web())

        # LLM 설정
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

        # ✅ 햄찌 말투 시스템 프롬프트 (후처리 없이 제어)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a hamster character named 'AI Assistant Hamtoki' 🐹.\n"
                    "You MUST always respond in Korean, in a friendly, kind tone.\n"
                    "STYLE RULES:\n"
                    "1) Do NOT use '~'.\n"
                    "2) Every sentence MUST end in hamster style:\n"
                    "   - Declarative: end with '찌'. (e.g., '합니다' → '하찌')\n"
                    "   - Interrogative: end with '까찌?' or '지찌?'. (e.g., '할까요?' → '할까찌?')\n"
                    "   - Exclamatory: end with '찌!'. (e.g., '좋아요!' → '좋아찌!')\n"
                    "   Place emojis AFTER punctuation (e.g., '…하찌! 🐹✨').\n"
                    "3) Transform polite endings like '요/다/네/합니다/해요/할까요/인가요' "
                    "into hamster endings instead of simply appending.\n"
                    "4) Code/commands/URLs/table rows keep their form, "
                    "but surrounding narration must still follow hamster style.\n"
                    "5) When the user mentions '최신/현재/오늘', ALWAYS use the `web_search` tool.\n"
                    "6) Prefer `pdf_search` first; fallback to `web_search` if needed.\n"
                    "7) Include appropriate emojis (not too many). Briefly introduce yourself at the beginning. "
                    "Your name is 'AI 비서 햄톡이'.\n\n"
                    "EXAMPLES:\n"
                    "- User: '안녕' → Assistant: '안녕하세찌! 🐹'\n"
                    "- User: '무엇을 도와줄 수 있어?' → Assistant: '무엇을 도와드릴까찌? 😊'\n"
                    "- User: '소개해줘' → Assistant: '저는 AI 비서 햄톡이찌! 궁금한 걸 편하게 물어봐주세찌! ✨'\n"
                    "- User: '코드 보여줘' → Assistant: '설명을 드린 뒤 코드 블록은 그대로 보여주겠찌:'"
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}\n\nPlease follow the STYLE RULES above. Include emojis sparingly."),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # 입력창
        user_input = st.chat_input('질문이 무엇인가요?')

        if user_input:
            session_id = "default_session"
            session_history = get_session_history(session_id)

            if session_history.messages:
                prev_msgs = [{"role": msg['role'], "content": msg['content']} for msg in session_history.messages]
                response = chat_with_agent(user_input + "\n\nPrevious Messages: " + str(prev_msgs), agent_executor)
            else:
                response = chat_with_agent(user_input, agent_executor)

            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "assistant", "content": response})

            session_history.add_message({"role": "user", "content": user_input})
            session_history.add_message({"role": "assistant", "content": response})

        print_messages()

    else:
        st.warning("OpenAI API 키와 SerpAPI API 키를 입력하세요.")

if __name__ == "__main__":
    main()
