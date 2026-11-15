import streamlit as st
from groq import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError

from agent import MODEL, ExamAgent

st.set_page_config(page_title="AI Examiner", page_icon="🎓")
st.title("🎓 AI Examiner — іспит з NLP")
st.write(
    "LLM-агент проводить усний іспит з курсу «Обробка природної мови». "
    "Студент вводить ім'я та email у формі на сайті, система перевіряє студента в базі, отримує теми через "
    "tool calling, опитує діалогом, виставляє оцінку 0–10 і зберігає результат."
)

with st.sidebar:
    st.header("Налаштування")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.caption(f"Модель: {MODEL}")
    if st.button("Почати заново"):
        st.session_state.clear()
        st.rerun()

if not api_key.strip():
    st.info("Введіть Groq API Key у бічній панелі, щоб почати.")
    st.stop()

if "agent" not in st.session_state:
    st.session_state.agent = ExamAgent(api_key.strip())
    st.session_state.chat = []
    st.session_state.started = False
    st.session_state.error = None
agent: ExamAgent = st.session_state.agent
agent.client.api_key = api_key.strip()


def api_call(fn, *args):
    """Виклик агента з обробкою помилок API. Повертає відповідь або None.
    Помилка зберігається в session_state, щоб не зникнути після st.rerun().
    """
    st.session_state.error = None
    try:
        with st.spinner("Екзаменатор думає..."):
            return fn(*args)
    except AuthenticationError:
        st.session_state.error = "Невірний Groq API Key."
    except RateLimitError:
        st.session_state.error = "Перевищено ліміт запитів. Спробуйте пізніше."
    except APIConnectionError:
        st.session_state.error = "Немає з'єднання з Groq."
    except APIStatusError as e:
        st.session_state.error = f"Помилка API (HTTP {e.status_code}). Спробуйте повторити запит."
    return None


# Крок 1: форма входу на сайті (ім'я та email)
if not st.session_state.started:
    st.subheader("Вхід на іспит")
    with st.form("login"):
        name = st.text_input("Ім'я та прізвище")
        email = st.text_input("Email")
        submitted = st.form_submit_button("Почати іспит", type="primary")
    if submitted:
        if not name.strip() or not email.strip():
            st.warning("Введіть ім'я та email.")
        else:
            try:
                reply = api_call(agent.begin, email.strip(), name.strip())
                if reply is not None:
                    st.session_state.started = True
                    st.session_state.chat.append(("assistant", reply))
                    st.rerun()
            except ValueError as e:  # студента немає в базі
                st.error(str(e))
    if st.session_state.error:
        st.error(st.session_state.error)
    st.stop()

# Крок 2: діалог
for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)

if st.session_state.error:
    st.error(st.session_state.error)
    if st.button("Повторити запит"):  # повтор без нового повідомлення студента
        reply = api_call(agent.step, None)
        if reply is not None:
            st.session_state.chat.append(("assistant", reply))
        st.rerun()

if agent.session.finished:
    st.success(f"Іспит завершено. Оцінка: {agent.score}/10")
elif prompt := st.chat_input("Ваша відповідь..."):
    st.session_state.chat.append(("user", prompt))
    reply = api_call(agent.step, prompt)
    if reply is not None:
        st.session_state.chat.append(("assistant", reply))
    st.rerun()
