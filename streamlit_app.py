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
from typing import List
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader, CSVLoader
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
# fallback용
# from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript
# import yt_dlp
# import tempfile, os

# .env 파일 로드
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 공통: 문서 → 벡터DB → (선택) 압축 리트리버
def _build_retriever_from_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200, compress: bool = True):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(documents)
    vector = FAISS.from_documents(splits, OpenAIEmbeddings())
    base = vector.as_retriever(search_kwargs={"k": 6})

    if not compress:
        return base

    # 쿼리 관련 핵심만 추출하는 압축 리트리버
    llm_for_compress = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    compressor = LLMChainExtractor.from_llm(llm_for_compress)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base)

# 웹페이지 RAG 툴
def create_website_rag_tool(urls: List[str]):
    urls = [u.strip() for u in urls if u and u.strip()]
    if not urls:
        return None
    loader = WebBaseLoader(urls)
    docs = loader.load()
    retriever = _build_retriever_from_documents(docs, compress=True)
    return create_retriever_tool(
        retriever,
        name="website_search",
        description="RAG over provided web pages. 공식 문서/블로그/위키 등 URL 근거 기반 검색에 사용."
    )

# 유튜브 자막 RAG 툴
def create_youtube_rag_tool(video_urls: list[str]):
    video_urls = [u.strip() for u in (video_urls or []) if u and u.strip()]
    if not video_urls:
        return None

    all_docs, errors = [], []

    def _load_with_langchain(url: str):
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,
            language=["ko", "en"],   # 우선 ko, 다음 en
            translation="ko",        # 번역자막 허용
        )
        return loader.load()

    def _load_with_transcript_api(url: str):
        # youtube_transcript_api 직접 호출
        vid = url.split("v=", 1)[1].split("&", 1)[0]
        tr_list = YouTubeTranscriptApi.list_transcripts(vid)

        # ko 우선 → en → en을 ko 번역
        try:
            tr = tr_list.find_transcript(["ko"])
        except Exception:
            tr = None
        if not tr:
            try:
                tr = tr_list.find_transcript(["en"])
            except Exception:
                tr = None
        fetched = None
        if tr:
            try:
                fetched = tr.fetch()
            except Exception:
                fetched = None

        if not fetched:
            # en → ko 번역 시도
            try:
                tr = tr_list.find_transcript(["en"]).translate("ko")
                fetched = tr.fetch()
            except Exception:
                fetched = None

        if not fetched:
            raise NoTranscriptFound(vid)

        text = "\n".join([i["text"] for i in fetched if i.get("text")])
        lang = (tr.language_code if hasattr(tr, "language_code") else "ko")
        return [Document(page_content=text, metadata={"source": url, "language": lang, "loader": "youtube_transcript_api"})]

    def _load_with_ytdlp(url: str):
        # 자동생성/업로더 제공 자막 파일을 내려받아 파싱
        with tempfile.TemporaryDirectory() as tmpdir:
            ydl_opts = {
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["ko", "en"],
                "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
                "quiet": True,
                "noprogress": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                vid = info["id"]

            # ko 우선 → en
            for lang in ["ko", "en"]:
                vtt = os.path.join(tmpdir, f"{vid}.{lang}.vtt")
                srt = os.path.join(tmpdir, f"{vid}.{lang}.srt")
                path = vtt if os.path.exists(vtt) else (srt if os.path.exists(srt) else None)
                if not path:
                    continue
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                # 매우 단순한 VTT/SRT 텍스트 추출
                lines = []
                for line in content.splitlines():
                    line = line.strip()
                    if not line or line.startswith("WEBVTT") or "-->" in line or line.isdigit():
                        continue
                    lines.append(line)
                text = "\n".join(lines)
                return [Document(page_content=text, metadata={"source": url, "language": lang, "loader": "yt_dlp"})]

        raise NoTranscriptFound(url)

    docs = []
    for url in video_urls:
        try:
            docs = _load_with_langchain(url)
        except Exception as e1:
            try:
                docs = _load_with_transcript_api(url)
            except Exception as e2:
                try:
                    docs = _load_with_ytdlp(url)
                except Exception as e3:
                    errors.append(f"{url} → {type(e3).__name__}: {e3}")
                    docs = []

        all_docs.extend(docs)

    if not all_docs:
        st.warning(
            "YouTube 자막을 불러오지 못했찌. 아래 원인을 확인해줘찌:\n\n- " + "\n- ".join(errors) if errors
            else "YouTube 자막이 없어 RAG에 사용할 문서를 만들 수 없었찌."
        )
        return None

    # 벡터DB/리트리버 구성 (기존과 동일)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(all_docs)
    vector = FAISS.from_documents(splits, OpenAIEmbeddings())
    retriever = vector.as_retriever(search_kwargs={"k": 6})

    return create_retriever_tool(
        retriever,
        name="youtube_search",
        description="RAG over YouTube transcripts (ko/en). 링크된 영상의 자막을 근거로 답변합니다."
    )


# FAQ CSV RAG 툴
def create_faq_csv_rag_tool(uploaded_csv_files):
    if not uploaded_csv_files:
        return None
    docs = []
    for f in uploaded_csv_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(f.read())
            path = tmp_file.name
        loader = CSVLoader(file_path=path, encoding="utf-8")
        docs.extend(loader.load())
    retriever = _build_retriever_from_documents(docs, chunk_size=800, chunk_overlap=100, compress=True)
    return create_retriever_tool(
        retriever,
        name="faq_search",
        description="RAG over uploaded FAQ CSV (Q/A, category, product sheet 등)."
    )

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

        # 웹페이지 RAG URL들 (줄바꿈 구분)
        urls_text = st.text_area("RAG 기반 웹페이지 URL", placeholder="https://docs.langchain.com\nhttps://openai.com/research",help="특정 제품 문서/블로그/위키를 벡터화해서 “출처 기반” 답변을 줘서 신뢰도가 높아지찌.")
        st.session_state["website_urls"] = [u.strip() for u in urls_text.splitlines() if u.strip()]

        # 유튜브 RAG URL들 (줄바꿈 구분)
        yt_text = st.text_area("RAG 기반 YouTube URL", placeholder="https://www.youtube.com/watch?v=XXXXXXXX", help="긴 영상도 자막을 조각 내서 질문에 맞는 부분만 찾아 요약해주니, 교육/리뷰/강의에 특히 쓸모 있찌.")
        st.session_state["youtube_urls"] = [u.strip() for u in yt_text.splitlines() if u.strip()]

        # FAQ CSV 업로드
        faq_csvs = st.file_uploader("FAQ CSV 업로드", type=["csv"], accept_multiple_files=True, key="faq_csv_uploader", help="내부 FAQ/고객응대 CSV를 바로 RAG에 얹어 실무 챗봇화하기 딱 좋찌.")
        st.session_state["faq_csvs"] = faq_csvs

        

    # ✅ 키 입력 확인
    if st.session_state["OPENAI_API"] and st.session_state["SERPAPI_API"]:
        os.environ['OPENAI_API_KEY'] = st.session_state["OPENAI_API"]
        os.environ['SERPAPI_API_KEY'] = st.session_state["SERPAPI_API"]

        # 도구 정의
        tools = []
        if pdf_docs:
            pdf_search = load_pdf_files(pdf_docs)
            tools.append(pdf_search)
        
        # 👉 새 RAG 툴들 추가
        website_tool = create_website_rag_tool(st.session_state.get("website_urls", []))
        if website_tool:
            tools.append(website_tool)

        youtube_tool = create_youtube_rag_tool(st.session_state.get("youtube_urls", []))
        if youtube_tool:
            tools.append(youtube_tool)

        faq_tool = create_faq_csv_rag_tool(st.session_state.get("faq_csvs", None))
        if faq_tool:
            tools.append(faq_tool)

        # 기존 웹 검색 툴
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
                    "... When the user references a YouTube link or asks about a video, use the `youtube_search` tool to ground your answer. ..."
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
