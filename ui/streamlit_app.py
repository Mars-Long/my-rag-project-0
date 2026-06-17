"""Streamlit UI for RAG0 — a minimal, clean chat interface.

Key fixes over the old ``web_app.py``:
- All operations go through the API (no direct SQLite imports).
- Uses ``/health`` endpoint for connection check (was ``/docs``).
- Proper error handling with user-visible messages.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src/ to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import requests

# Configuration
API_BASE = "http://127.0.0.1:7861"


# ---------------------------------------------------------------------------
# Helper: API client
# ---------------------------------------------------------------------------
class APIClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def check_health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def list_kbs(self) -> list[dict]:
        r = requests.get(f"{self.base_url}/knowledge-bases", timeout=10)
        r.raise_for_status()
        return r.json()

    def create_kb(self, name: str, description: str = "") -> dict:
        r = requests.post(
            f"{self.base_url}/knowledge-bases",
            data={"name": name, "description": description},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def delete_kb(self, name: str) -> dict:
        r = requests.delete(f"{self.base_url}/knowledge-bases/{name}", timeout=30)
        r.raise_for_status()
        return r.json()

    def upload_docs(self, kb_name: str, files: list) -> list:
        from io import BytesIO

        uploaded = []
        for f in files:
            r = requests.post(
                f"{self.base_url}/knowledge-bases/{kb_name}/documents",
                files={"files": (f.name, f.getvalue(), f.type)},
                data={"name": kb_name},
                timeout=120,
            )
            r.raise_for_status()
            uploaded.append(r.json())
        return uploaded

    def chat(self, query: str, kb_name: str, history: list | None = None) -> requests.Response:
        return requests.post(
            f"{self.base_url}/chat",
            json={
                "query": query,
                "knowledge_base_name": kb_name,
                "history": history or [],
                "top_k": 5,
                "stream": True,
                "return_docs": True,
            },
            stream=True,
            timeout=120,
        )


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="RAG0", page_icon="📚", layout="wide")

api = APIClient(API_BASE)

# Connection check
if "api_ok" not in st.session_state:
    st.session_state.api_ok = api.check_health()

if not st.session_state.api_ok:
    st.error(f"无法连接到 API 服务器 ({API_BASE})。请先启动后端：`python -m rag0 serve`")
    if st.button("重试连接"):
        st.session_state.api_ok = api.check_health()
        st.rerun()
    st.stop()

# ---- Sidebar: KB management ----
with st.sidebar:
    st.header("📚 知识库管理")

    # List KBs
    if st.button("刷新列表"):
        st.session_state.kbs = api.list_kbs()
    if "kbs" not in st.session_state:
        st.session_state.kbs = api.list_kbs()

    kb_names = [kb["name"] for kb in st.session_state.kbs]
    selected_kb = st.selectbox("选择知识库", kb_names, key="kb_selector")

    # Create new KB
    with st.expander("新建知识库"):
        new_kb_name = st.text_input("名称")
        if st.button("创建") and new_kb_name:
            try:
                api.create_kb(new_kb_name)
                st.success(f"知识库 '{new_kb_name}' 已创建")
                st.session_state.kbs = api.list_kbs()
            except Exception as e:
                st.error(str(e))

    # Upload files
    if selected_kb:
        with st.expander("上传文档"):
            uploaded = st.file_uploader(
                "选择文件", accept_multiple_files=True,
                type=["pdf", "docx", "txt", "md"],
            )
            if st.button("开始索引") and uploaded:
                with st.spinner("索引中..."):
                    try:
                        results = api.upload_docs(selected_kb, uploaded)
                        st.success(f"成功索引 {len(results)} 个文件")
                        st.session_state.kbs = api.list_kbs()
                    except Exception as e:
                        st.error(str(e))

        # Delete KB
        if st.button("删除此知识库", type="secondary"):
            try:
                api.delete_kb(selected_kb)
                st.session_state.kbs = api.list_kbs()
                st.rerun()
            except Exception as e:
                st.error(str(e))

# ---- Chat area ----
st.title("RAG0 — 知识库对话")

if not selected_kb:
    st.info("请在侧边栏选择一个知识库开始对话")
    st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            response = api.chat(prompt, selected_kb)
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ")
                if data == "[DONE]":
                    break
                try:
                    import json
                    payload = json.loads(data)
                    content = payload.get("content", "")
                    full_response += content
                    placeholder.markdown(full_response + "▌")
                except Exception:
                    pass
        except Exception as e:
            full_response = f"错误: {e}"

        placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
