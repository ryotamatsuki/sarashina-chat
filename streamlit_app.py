import requests
import streamlit as st


st.set_page_config(
    page_title="Sarashina Chat",
    page_icon="💬",
    layout="centered",
)

st.title("Sarashina2.2-3B-Instruct")
st.caption(
    "SB Intuitionsの日本語LLMを使用した試験用チャットです。"
)

MODAL_URL = st.secrets["MODAL_URL"]
MODAL_KEY = st.secrets["MODAL_KEY"]
MODAL_SECRET = st.secrets["MODAL_SECRET"]

if "messages" not in st.session_state:
    st.session_state.messages = []


# 過去の会話を表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("メッセージを入力してください")

if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Sarashinaが回答を生成しています..."):
            try:
                response = requests.post(
                    MODAL_URL,
                    headers={
                        "Modal-Key": MODAL_KEY,
                        "Modal-Secret": MODAL_SECRET,
                        "Authorization": f"Bearer {MODAL_KEY}.{MODAL_SECRET}",
                        "Content-Type": "application/json",
                    },
                    json={"messages": st.session_state.messages},
                    timeout=300,
                )

                response.raise_for_status()
                result = response.json()

                if "error" in result:
                    answer = f"エラー: {result['error']}"
                else:
                    answer = result["answer"]

            except Exception as exc:
                answer = f"バックエンドとの通信に失敗しました。 {exc}"

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


with st.sidebar:
    st.header("設定")

    if st.button("会話をリセット"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption(
        "個人情報・機密情報・未公開情報は入力しないでください。"
    )
