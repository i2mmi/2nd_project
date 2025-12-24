
import streamlit as st
import json
import os
import numpy as np
import re
from openai import OpenAI

# ==========================================
# 1. 설정 및 상수 정의
# ==========================================
DATA_PATH = "data/card_benefits_with_id.json"
EMBED_MODEL = "text-embedding-3-small"
BASE_CHAT_MODEL = "gpt-3.5-turbo"

# 캐시 경로 설정
ARTIFACT_DIR = "artifacts"
EMB_CACHE_PATH = os.path.join(ARTIFACT_DIR, "card_emb_cache.npz")
DOCS_CACHE_PATH = os.path.join(ARTIFACT_DIR, "card_docs_cache.json")

os.makedirs(ARTIFACT_DIR, exist_ok=True)

st.set_page_config(layout="wide", page_title="카드고릴라 RAG 챗봇")

# ==========================================
# 2. 유틸리티 함수 (전처리, 임베딩 등)
# ==========================================

def _clean_text(s: str) -> str:
    s = (s or "").replace("\n", " ").replace("\t", " ")
    return re.sub(r'\s+', ' ', s).strip()

def make_card_context_text(card):
    """카드 정보를 LLM이 읽기 좋은 텍스트로 변환"""
    summary_parts = []
    current_len = 0
    # 혜택 요약 로직 (간소화 버전)
    for b in card.get('benefits', []):
        cat = _clean_text(b.get('category'))
        detail = _clean_text(b.get('detail'))
        if len(detail) > 45: detail = detail[:42] + ".."
        line = f"[{cat}] {detail}"
        if current_len + len(line) > 500: break
        summary_parts.append(line)
        current_len += len(line)
    
    summary = " / ".join(summary_parts)
    return (
        f"card_id: {card.get('card_id')}\n"
        f"카드명: {card.get('card_name')}\n"
        f"카드사: {card.get('company')}\n"
        f"연회비: {card.get('annual_fee')}\n"
        f"혜택요약: {summary}\n"
    )

def embed_many(client, texts, model=EMBED_MODEL):
    """임베딩 생성 (배치 처리)"""
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            resp = client.embeddings.create(model=model, input=batch)
            all_embeddings.extend([d.embedding for d in resp.data])
        except Exception as e:
            st.error(f"임베딩 오류: {e}")
            return np.array([])
            
    return np.array(all_embeddings, dtype=np.float32)

def l2_normalize(X):
    norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norm

# ==========================================
# 3. 데이터 로드 및 임베딩 캐싱 (핵심!)
# ==========================================

@st.cache_resource
def load_data_and_embeddings(api_key):
    """
    JSON 데이터와 임베딩을 로드합니다.
    임베딩 파일이 없으면 생성하여 저장합니다.
    """
    if not api_key: return None, None, None

    # 1. JSON 카드 데이터 로드
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        cards = json.load(f)
    
    # 2. 임베딩 캐시 확인
    if os.path.exists(EMB_CACHE_PATH) and os.path.exists(DOCS_CACHE_PATH):
        with open(DOCS_CACHE_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)
        emb = np.load(EMB_CACHE_PATH)["emb"]
        return cards, docs, emb
    
    # 3. 캐시 없으면 새로 생성 (최초 1회)
    client = OpenAI(api_key=api_key)
    docs = []
    for card in cards:
        # 메타 정보 청크
        base_text = f"{card.get('card_name')} {card.get('company')} 연회비:{card.get('annual_fee')}"
        docs.append({"card_id": str(card['card_id']), "text": base_text, "type": "meta"})
        
        # 혜택 정보 청크 (Chunking)
        for b in card.get('benefits', []):
            chunk = f"카드명:{card.get('card_name')} | 혜택:[{b.get('category')}] {b.get('detail')}"
            docs.append({"card_id": str(card['card_id']), "text": chunk, "type": "benefit"})
    
    texts = [d["text"] for d in docs]
    
    with st.spinner("🚀 최초 실행: 벡터 임베딩 생성 중... (시간이 좀 걸립니다)"):
        emb = embed_many(client, texts)
        emb = l2_normalize(emb)
    
    # 저장
    with open(DOCS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)
    np.savez_compressed(EMB_CACHE_PATH, emb=emb)
    
    return cards, docs, emb

# ==========================================
# 4. RAG 검색 및 생성 로직
# ==========================================

def retrieve_topk(client, query, docs, doc_emb, top_k=15): # 15개로 늘림!
    """검색 로직"""
    q_emb = embed_many(client, [query])[0]
    q_emb = l2_normalize(q_emb.reshape(1, -1))[0]
    
    scores = doc_emb @ q_emb
    top_idx = np.argsort(-scores)[:top_k]
    
    return [docs[i] for i in top_idx], scores[top_idx]

def recommend_rag_stream(client, cards, docs, doc_emb, persona_text, user_question):
    # 1. 키워드 추출 (Query Transformation)
    kw_prompt = f"""
    사용자의 질문과 페르소나를 분석해 카드 검색에 필요한 '핵심 키워드'만 콤마로 구분해 뽑아줘.
    [페르소나]: {persona_text}
    [질문]: {user_question}
    [예시]: 공과금 할인, 통신비 자동이체, 편의점
    """
    kw_res = client.chat.completions.create(model=BASE_CHAT_MODEL, messages=[{"role":"user", "content": kw_prompt}])
    keyword = kw_res.choices[0].message.content
    print(f"🔍 추출된 키워드: {keyword}") # 터미널 로그 확인용

    # 2. 검색 (Retrieval)
    retrieved_chunks, scores = retrieve_topk(client, keyword, docs, doc_emb, top_k=15)
    
    # 3. 중복 제거 (Filtering) - 상위 3개 카드 선정
    seen_ids = set()
    candidate_ids = []
    for chunk in retrieved_chunks:
        cid = chunk['card_id']
        if cid not in seen_ids:
            candidate_ids.append(cid)
            seen_ids.add(cid)
        if len(candidate_ids) >= 3:
            break
            
    # 4. 전체 컨텍스트 로드 (Context Injection)
    # 검색된 ID로 원본 카드 데이터를 찾음
    candidates = [c for c in cards if str(c['card_id']) in candidate_ids]
    context_text = "\n---\n".join([make_card_context_text(c) for c in candidates])
    
    # 5. 답변 생성
    sys_prompt = f"""
    너는 데이터 기반 카드 추천 전문가다.
    아래 [검색된 카드 정보]만을 근거로 사용하여 질문에 답하라.
    
    [사용자 프로필]
    {persona_text}
    
    [검색된 카드 정보]
    {context_text}
    
    [작성 규칙]
    1. 추천 카드 1~2개를 선정하고 이유를 상세히 설명하라.
    2. 카드 정보에 없는 내용은 절대 지어내지 말라.
    3. 추천 사유에 혜택의 구체적인 수치(예: 5%, 1천원 등)를 인용하라.
    """
    
    response = client.chat.completions.create(
        model=BASE_CHAT_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_question}
        ],
        stream=True
    )
    return response, keyword # 키워드도 같이 반환해서 화면에 보여주면 좋음

# ==========================================
# 5. Streamlit UI
# ==========================================

if "section" not in st.session_state: st.session_state.section = "chatbot"
if "messages" not in st.session_state: st.session_state.messages = []

# 사이드바
with st.sidebar:
    st.title("💳 카드고릴라 RAG")
    api_key = st.text_input("OpenAI API Key", type="password")
    
    # 데이터 로드 (API 키가 있어야 실행됨)
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        cards_data, docs_data, emb_data = load_data_and_embeddings(api_key)
        if cards_data:
            st.success(f"데이터 로드 완료! (카드 {len(cards_data)}장)")
    else:
        st.warning("API Key를 입력해주세요.")

    st.divider()
    persona_input = st.text_area("내 프로필 (검색에 반영됨)", value="30대 직장인, 자취생, 월 100만원 사용", height=100)
    
    st.divider()
    if st.button("🤖 챗봇"): st.session_state.section = "chatbot"
    if st.button("📊 구현 과정"): st.session_state.section = "process"

# 메인 화면
if st.session_state.section == "chatbot":
    st.title("🤖 AI 카드 추천 챗봇")

    # 채팅 기록 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요"):
        if not api_key:
            st.error("API Key가 필요합니다!")
            st.stop()
            
        client = OpenAI(api_key=api_key)
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 답변 생성
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("분석 중..."):
                try:
                    # RAG 호출
                    stream, keyword = recommend_rag_stream(client, cards_data, docs_data, emb_data, persona_input, prompt)
                    
                    # (선택) 검색된 키워드 살짝 보여주기
                    st.caption(f"🔍 검색 키워드: {keyword}")

                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    st.error(f"에러 발생: {e}")
                    full_response = "오류가 발생했습니다."
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})


elif st.session_state.section == "process":
    st.title("구현 과정")

    tab1, tab2, tab3 = st.tabs(
        ["크롤링", "프롬프트 설계", "RAG 파이프라인"]
    )

    # =====================
    # step1. 크롤링
    # =====================
    with tab1:
        st.header("🕷️ 데이터 수집 (Crawling)")
        st.markdown("Selenium을 활용한 동적 웹페이지 크롤링 프로세스입니다.")
        st.markdown("---")

        # 1. 개요 섹션 (좌우 분할)
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                st.subheader("🎯 수집 대상")
                st.info("""
                **카드고릴라 웹사이트**
                - 🏆 TOP 100 차트 목록
                - 💳 카드 상세 페이지 (신용/체크)
                """)

        with col2:
            with st.container(border=True):
                st.subheader("🔄 작업 흐름 (Workflow)")
                st.markdown("""
                1. **URL 수집:** 목록 페이지에서 상세 링크(`href`) 추출
                2. **상세 접근:** 각 URL로 직접 이동 (`driver.get`)
                3. **데이터 파싱:** 혜택, 연회비, 브랜드 등 정보 수집
                4. **DB 구축:** 체크(0)/신용(1) 구분 후 JSON 저장
                """)

        st.divider()

        # 2. 설계 의도 섹션 (문제 vs 해결)
        st.subheader("💡 설계 의도 (Problem & Solution)")
        st.markdown("교육 과정에서 배운 방식과 실제 현업 사이트의 차이를 극복한 과정입니다.")

        p_col1, p_col2 = st.columns(2)
        
        with p_col1:
            st.warning("**⛔ 문제점: Click 이벤트 불안정**")
            st.caption("Selenium `element.click()` 실패")
            st.write("""
            - 동적으로 생성되는 리스트 구조
            - 화면 렌더링 속도 차이로 인한 `ElementNotInteractable` 에러 빈번 발생
            """)
        
        with p_col2:
            st.success("**✅ 해결책: URL 직접 접근 (Direct Access)**")
            st.caption("`get_attribute('href')` 활용")
            st.write("""
            - 상세 페이지 URL 리스트를 먼저 확보
            - `driver.get(url)`로 직접 이동하여 안정성 100% 확보
            - 크롤링 속도 및 에러 핸들링 용이
            """)

        st.divider()

        # 3. 핵심 함수 설명 (카드 UI 형태)
        st.subheader("💻 핵심 코드 하이라이트")

        # 함수 1: human_sleep
        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("#### 1. 봇 탐지 회피")
                st.caption("`human_sleep`")
                st.write("고정된 시간(`sleep(1)`)은 봇으로 탐지될 위험이 큽니다. 랜덤 딜레이를 주어 사람처럼 행동하게 합니다.")
            with c2:
                st.code("""
                import random, time
                
                def human_sleep(min_sec=0, max_sec=3):
                    # 0~3초 사이의 랜덤한 실수(float) 시간만큼 대기
                    time.sleep(random.uniform(min_sec, max_sec))
                                """, language="python")

        # 함수 2: get_text_or_empty
        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("#### 2. 에러 방지 처리")
                st.caption("`get_text_or_empty`")
                st.write("온라인 전용 카드 등 일부 정보가 없는 경우에도 크롤러가 멈추지 않도록 빈 문자열(`""`)을 반환합니다.")
            with c2:
                st.code("""
                def get_text_or_empty(driver, by, selector):
                    try:
                        return driver.find_element(by, selector).text.strip()
                    except:
                        # 요소가 없어도 에러를 내지 않고 빈 값 반환
                        return ""
                                """, language="python")

        # 함수 3: get_card_urls
        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("#### 3. URL 리스트 추출")
                st.caption("`get_card_urls`")
                st.write("클릭 대신 `href` 속성값을 리스트로 추출하여 이후 작업의 기반을 마련합니다.")
            with c2:
                st.code("""
                def get_card_urls(driver, limit=None):
                    # CSS Selector로 카드 링크 요소 찾기
                    cards = driver.find_elements(By.CSS_SELECTOR, "a.card_link")
                    # href 속성만 추출하여 리스트로 저장
                    urls = [card.get_attribute("href") for card in cards[:limit]]
                    return urls
                                """, language="python")

        # 함수 4: get_card_detail
        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("#### 4. 디테일 파싱 (Brand)")
                st.caption("`get_card_detail`")
                st.write("**[Parsing Tip]** 브랜드 텍스트가 `VISAAMEX` 처럼 붙어 나오는 문제를 해결하기 위해 `span` 태그 단위로 쪼개서 합쳤습니다.")
            with c2:
                st.code("""
                def get_card_detail(driver, url):
                    driver.get(url)
                    human_sleep() # 사람인 척 대기
                
                    # 브랜드가 여러 개일 경우 텍스트 겹침 방지
                    brands = driver.find_elements(By.CSS_SELECTOR, "dd.c_brand span")
                    brand_text = " ".join([b.text for b in brands]) # "VISA AMEX"
                
                    return {"url": url, "brand": brand_text}
                                """, language="python")

# =====================
    # step2. 프롬프트 설계 
    # =====================
    with tab2:
        st.header("🧪 프롬프트 설계 기준 및 실험")
        st.markdown("---")

        # 1. 설계 기준 섹션
        st.subheader("1. 프롬프트 설계 규칙")
        st.write("저희 팀은 3가지 규칙을 정의했습니다.")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            # st.info: 파란색 박스 (정보/정의)
            st.info("🕵️ 단순 챗봇이 아닌 **데이터 기반 전문가**로 역할을 정의했습니다.")
        
        with col2:
            # st.warning: 노란색 박스 (주의/제한)
            st.warning("🚫 제공된 **Context** 외의 외부 지식 사용을 금지하여 **Hallucination을 방지**합니다.")
        
        with col3:
            # st.success: 초록색 박스 (해결책/성공요소)
            st.success("💬 추천 사유 작성 시 **'근거 문장'을 반드시 인용** 하도록 강제했습니다.")

        st.markdown("---")

        # 2. 단계별 진화 섹션 (탭 안에 또 탭을 넣어 비교)
        st.subheader("2. 모델 고도화 과정 (Evolution)")
        
        # Nested Tabs (내부 탭)
        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["① Base Model", "② Prompt Engineering", "③ RAG (Final)"])

        with sub_tab1:
            st.markdown("#### 🔹 Base Model: 단순 정보 주입")
            st.markdown("별도의 시스템 프롬프트 없이, 상위 12개의 카드 정보만 컨텍스트로 제공했습니다.")
            st.code("""
            # 프롬프트 구조
            [사용자 페르소나] ...
            [사용자 질문] ...
            [카드 후보 목록] (상위 12개 고정)
            
            # ❌ 한계점
            - 검색 기능 부재로 제한된 카드만 볼 수 있음
            - 혜택을 두루뭉술하게 설명하며, 없는 혜택을 지어내기도 함
                        """, language="markdown")

        with sub_tab2:
            st.markdown("#### 🔹 Prompt Engineering: 제약 조건 강화")
            st.markdown("Base 모델에 **System Prompt**를 추가하여 논리력과 근거 인용을 강제했습니다.")
            st.code("""
            [System Prompt]
            너는 카드 추천 전문가다.
            규칙:
            1) 추천 이유는 '주요혜택요약'에서 인용한 문장을 포함해야 한다.
            2) 연회비/조건이 사용자 니즈와 충돌하면 불리하게 반영하라.
            3) 없는 혜택은 절대 지어내지 말라.
                        """, language="python")

        with sub_tab3:
            st.markdown("#### 🔹 RAG Model: 검색어 최적화 (Query Transformation)")
            st.markdown("단순 검색의 낮은 유사도(0.4)를 극복하기 위해 **질문 변환 프롬프트**를 도입했습니다.")
            
            c1, c2 = st.columns(2)
            with c1:
                st.error("Before (Raw Query)")
                st.write("'저는 30대 직장인이고 자취를 하며 주말에는...'")
                st.caption("👉 불필요한 서술어가 노이즈로 작용하여 검색 정확도 하락")
            with c2:
                st.success("After (Keyword Extraction)")
                st.code("공과금 할인, 통신비 자동이체, 편의점 혜택")
                st.caption("👉 핵심 키워드만 추출하여 검색 유사도 0.55+ 달성")

        st.markdown("---")

        # 3. 실험 결론 섹션
        st.subheader("3. 실험 결론")
        st.markdown("""
            ### 👤 입력 Persona
            
            **1. 유형 (Title)**
            > **30대 사회초년생(자취) – 통신비·공과금 자동이체 중심**
        
            **2. 상세 내용 (Profile)**
            ```text
            연령대: 30대 초중반
            직업군: 사회초년생 / 주거형태: 자취
            소비유형: 고정비 중심
            주요지출: 휴대폰 요금, 전기/가스/수도 요금
            결제방식: 자동이체
            중요조건: 통신비/공과금 할인, 전월실적 조건 단순, 연회비 낮음
            비선호: 복잡한 조건, 높은 전월실적
            ```
        
            **3. 사용자 질문 (Question)**
            "공과금과 통신비 자동이체만으로 확실한 할인을 받을 수 있는 카드를 추천해주세요."
        
            ---
            **🔑 추출된 Keyword**
            `🔍 [RAG] 최적화된 검색어: 공과금 할인, 통신비 할인, 자동이체, 연회비 없음`
            """)
        
        st.image("img/comp.png", 
         caption="모델 응답 비교 (Base vs Prompt Eng vs RAG)", 
         use_container_width=True)

    # ===================== 세은/정민
    # step3. RAG 파이프라인
    # ===================== 내용은 자율적으로 변경하셔도 됩니다.
    # =====================
    # step3. RAG 파이프라인
    # =====================
    with tab3:
        st.header("⚙️ RAG 시스템 아키텍처 (Pipeline)")
        st.markdown("데이터 수집부터 답변 생성까지의 전체 데이터 흐름도입니다.")
        
        # 1. 아키텍처 다이어그램 (Mermaid)
        st.markdown("#### 1. 전체 파이프라인 흐름도")
        
        # Mermaid로 흐름 시각화 (URL 방식 활용)
        # [수정 완료] I 노드의 텍스트를 '최종 카드 추천'으로 변경했습니다.
        mermaid_code = """
        graph TD
            subgraph Data_Indexing [데이터 구축 단계]
                A[카드 원본 데이터] -->|혜택 단위 청킹| B(Text Chunks)
                B -->|Embedding| C[(Vector DB / Cache)]
            end

            subgraph Retrieval_Process [검색 단계]
                D[사용자 질문] -->|Query Transformation| E[핵심 키워드 추출]
                E -->|Cosine Similarity| F{Vector DB 검색}
                F -->|Top-15 Fetch| G[관련 혜택 조각들]
                G -->|Deduplication| H[카드 단위 중복 제거]
                H -->|Select Top-3| I[최종 카드 추천] 
            end

            subgraph Generation_Process [생성 단계]
                I -->|Context Injection| J[LLM 프롬프트 구성]
                J -->|GPT-3.5-turbo| K[최종 추천 답변]
            end
            
            style Data_Indexing fill:#f9f9f9,stroke:#333,stroke-width:2px
            style Retrieval_Process fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
            style Generation_Process fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
        """
        # Mermaid 렌더링 (mermaid.ink API 사용)
        import base64
        mermaid_url = f"https://mermaid.ink/img/{base64.urlsafe_b64encode(mermaid_code.encode('utf-8')).decode('ascii')}"
        st.image(mermaid_url, caption="RAG System Architecture Flowchart", use_container_width=True)

        st.divider()

        # 2. 단계별 상세 설명
        st.markdown("#### 2. 단계별 핵심 로직 (Key Components)")

        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                st.markdown("##### 🗂️ 1. Vector DB & Chunking")
                st.caption("데이터 저장소 구축")
                st.markdown("""
                - **혜택 단위 청킹 (Benefit Chunking):**
                  카드 1장을 통째로 저장하지 않고, `혜택 1개 = 문서 1개`로 쪼개어 저장했습니다.
                  > *효과: "편의점" 검색 시 편의점 혜택이 있는 카드만 정확히 타격 가능*
                - **Batch Embedding:**
                  OpenAI API의 토큰 제한을 피하기 위해 데이터를 100개씩 나누어 처리했습니다.
                """)

        with col2:
            with st.container(border=True):
                st.markdown("##### 🔍 2. Advanced Retriever")
                st.caption("검색 품질 고도화")
                st.markdown("""
                - **Query Transformation:**
                  사용자의 긴 질문을 그대로 쓰지 않고, LLM을 통해 **'검색용 키워드'**만 추출했습니다.
                  > *결과: 유사도 0.4 → 0.6 이상 상승*
                - **Post-Filtering (Re-ranking):**
                  Top-15개의 혜택 조각을 가져온 뒤, 중복된 카드를 제거하고 **상위 3개의 유니크한 카드**만 선별했습니다.
                """)

        st.markdown("---")

        # 3. LLM 생성부 설명
        st.markdown("#### 3. LLM 답변 생성 (Generation)")
        st.info("""
        **📝 Context Injection (문맥 주입)**
        검색된 카드들에 대해서는 요약된 정보가 아닌 **'전체 상세 정보'**를 다시 로드하여 프롬프트에 주입합니다.
        이를 통해 LLM은 풍부한 정보를 바탕으로 **"왜 이 카드를 추천했는지"** 구체적인 근거를 들어 설명할 수 있습니다.
        """)
    
    # 세은/정민 ===================== 내용 추가를 원하시면 자율적으로 변경하셔도 됩니다.
    # 마지막까지 화이팅하시죠! 고생많으셨습니다.
