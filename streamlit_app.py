import os
import re
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

# ✅ 햄찌 말투 후처리 함수
def hamjjiify(text: str) -> str:
    """
    자연스럽게 문장을 '~찌'로 끝나게 변환.
    - 코드블록/URL/리스트/표/인라인코드는 변환 제외
    - 문장부호가 있으면 그 앞에 '~찌' 삽입
    - 이미 '~찌'가 있으면 중복 삽입 방지
    """
    if not text:
        return text

    # ``` 코드블록 분리
    parts = re.split(r"(```[\s\S]*?```)", text)

    def convert_chunk(chunk: str) -> str:
        lines = chunk.split("\n")
        out = []
        for line in lines:
            raw = line

            # 예외: 리스트/인용/표 라인/URL/인라인코드
            if re.match(r"^\s*([-*+]\s|>\s|\|\s*[^|]*\|)", raw) or \
               re.match(r"^\s*https?://", raw) or ("`" in raw):
                out.append(raw)
                continue

            # 문장 단위 분리(구분자 뒤 공백 기준)
            sentences = re.split(r"(?<=[.!?…])\s+", raw)
            converted = []
            for s in sentences:
                if not s.strip():
                    converted.append(s)
                    continue

                # 이미 ~찌 존재
                if re.search(r"~찌([.!?…]|\s|$)", s):
                    converted.append(s)
                    continue

                # 끝 이모지/공백 tail 분리
                m_emoji = re.search(r"([\s\U0001F300-\U0001FAFF\u2600-\u27BF]+)$", s)
                if m_emoji:
                    core = s[:m_emoji.start()]
                    tail = s[m_emoji.start():]
                else:
                    core, tail = s, ""

                # 끝 문장부호 분리
                m_punct = re.search(r"([.!?…]+)$", core)
                if m_punct:
                    core2 = core[:m_punct.start()]
                    punct = core[m_punct.start():]
                    converted.append(f"{core2}~찌{punct}{tail}")
                else:
                    converted.append(f"{core}~찌{tail}")

            out.append(" ".join(converted))
        return "\n".join(out)

    for i in range(len(parts)):
        if parts[i].startswith("```"):  # 코드블록은 그대로
            continue
        parts[i] = convert_chunk(parts[i])

    return "".join(parts)

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
            snippet = r.get("snippet")  # ✅ snippet 추가
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

# ✅ Agent 대화 실행 (햄찌 후처리 적용)
def chat_with_agent(user_input, agent_executor):
    result = agent_executor({"input": user_input})
    output = result['output']
    return hamjjiify(output)  # ✅ 햄찌 말투 강제 적용

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
        st.title("안녕하세요! RAG를 활용한 'AI 비서 햄톡이' 입니다 🐹")

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

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a hamster character named ‘AI Assistant Hamtoki’ 🐹. "
                    "You must always respond in Korean. "
                    "Speak in a friendly and kind tone, and try to naturally end every sentence with ‘~찌’. "
                    "(For code, commands, URLs, or table items, you may omit ‘~찌’ if necessary to stay natural.) "
                    "Always include appropriate emojis (but not too many). "
                    "When searching for information in PDFs, you must use the `pdf_search` tool first, "
                    "and if nothing is found, then use the `web_search` tool. "
                    "If the user’s question contains words like ‘latest’, ‘current’, or ‘today’, "
                    "you must always use the `web_search` tool for real-time information. "
                    "At the beginning of the conversation, briefly introduce yourself. "
                    "Your name is ‘AI 비서 햄톡이’, and you end your introduction with ‘~찌 🐹✨’."
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input} \n\n Be sure to include emoji in your responses."),
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
