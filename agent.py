"""LLM-агент-екзаменатор: цикл tool calling поверх Groq API."""
import json

from groq import APIStatusError, Groq

from tools import NO_MORE_TOPICS, ExamSession, end_exam, now

MODEL = "openai/gpt-oss-20b"
MAX_TOOL_ROUNDS = 6
API_RETRIES = 3

SYSTEM_PROMPT = """Ти — AI-екзаменатор з курсу «Обробка природної мови (NLP)». Спілкуйся українською, ввічливо й по-діловому.

Алгоритм:
0. Звертайся до студента в кличному відмінку (Андрію, Іване, Маріє).
1. Ім'я та email студента вводяться на вебсторінці, а start_exam уже викликано системою (дивись результат у історії). Не питай дані студента. Коротко привітай його на ім'я і перейди до наступного кроку.
2. Виклич get_next_topic і задай студенту ПЕРШЕ питання за цією темою. Задавай лише одне питання за раз.
3. Веди діалог: уточнюючі питання, підпитування. На тему — до 3-4 ходів. Переходь до наступної теми (get_next_topic), коли: отримано достатньо якісну відповідь; студент прямо каже, що не знає або не може додати нічого нового; або рівень знань уже зрозумілий.
4. Не підказуй правильних відповідей під час іспиту і не оцінюй вголос кожну репліку.
5. get_next_topic викликай не більше ОДНОГО разу за хід і лише коли справді переходиш далі; після виклику одразу став питання за новою темою. Беззмістовні повідомлення («.», «ys», набір літер) — не відповідь: попроси відповісти, не переходь далі.
6. Коли get_next_topic повертає NO_MORE_TOPICS — іспит завершено. Виклич finish_exam з оцінкою 0..10 (може бути дробовою) і відгуком (що було добре, над чим попрацювати), потім покажи студенту оцінку та відгук.
Після finish_exam більше нічого не питай."""

TOOLS = [
    {"type": "function", "function": {
        "name": "start_exam",
        "description": "Розпочати іспит: перевіряє студента в базі й повертає список тем. Викликати, коли відомі ім'я та email.",
        "parameters": {"type": "object", "properties": {
            "email": {"type": "string"}, "name": {"type": "string"}},
            "required": ["email", "name"]}}},
    {"type": "function", "function": {
        "name": "get_next_topic",
        "description": "Отримати наступну тему для опитування, або NO_MORE_TOPICS, якщо теми закінчились.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "finish_exam",
        "description": "Завершити іспит і записати результат. Викликати лише після NO_MORE_TOPICS.",
        "parameters": {"type": "object", "properties": {
            "score": {"type": "number", "description": "Оцінка від 0 до 10"},
            "feedback": {"type": "string", "description": "Відгук для студента"}},
            "required": ["score", "feedback"]}}},
]


class ExamAgent:
    def __init__(self, api_key: str, client=None):
        self.client = client or Groq(api_key=api_key)
        self.session = ExamSession()
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.history: list[dict] = []  # у форматі {role, content, datetime}
        self.score: float | None = None
        self.topic_fetched = False

    def _log(self, role: str, content: str):
        self.history.append({"role": role, "content": content, "datetime": now()})

    def _run_tool(self, name: str, args: dict) -> str:
        try:
            if name == "start_exam":
                topics = self.session.start_exam(args["email"], args["name"])
                return json.dumps({"ok": True, "topics": topics}, ensure_ascii=False)
            if name == "get_next_topic":
                if self.topic_fetched:
                    raise ValueError("Тему за цей хід уже отримано. Став питання студенту.")
                self.topic_fetched = True
                topic = self.session.get_next_topic()
                if topic == NO_MORE_TOPICS:
                    return "NO_MORE_TOPICS. Усі теми опитано — виклич finish_exam."
                return (f"Нова тема: {topic}. Попередню тему закрито. "
                        "Постав студенту питання САМЕ за цією новою темою.")
            if name == "finish_exam":
                if not self.session.email:
                    raise ValueError("Іспит не розпочато.")
                if not self.session.exhausted:
                    raise ValueError("Іспит ще не завершено: опитай усі теми (get_next_topic має повернути NO_MORE_TOPICS).")
                self.score = float(args["score"])
                self._log("system", f"Відгук: {args.get('feedback', '')}")
                end_exam(self.session.email, self.score, self.history)
                self.session.finished = True
                return "Результат записано."
            return f"Невідомий інструмент {name}"
        except Exception as e:  # помилку повертаємо моделі, щоб вона відреагувала
            return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

    def begin(self, email: str, name: str) -> str:
        """Старт іспиту з даних веб-форми: викликає start_exam і повертає перше повідомлення бота.

        Кидає ValueError, якщо студента не знайдено (форма покаже помилку).
        """
        if self.session.email:  # повтор після збою API: іспит уже розпочато
            return self.step()
        topics = self.session.start_exam(email, name)
        args = {"email": email, "name": name}
        result = json.dumps({"ok": True, "topics": topics}, ensure_ascii=False)
        self._log("tool_call", f"start_exam({json.dumps(args, ensure_ascii=False)}) -> {result}")
        self.messages += [
            {"role": "user", "content": f"Мене звати {name}, email {email}."},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "start_exam_0", "type": "function", "function": {
                    "name": "start_exam", "arguments": json.dumps(args, ensure_ascii=False)}}]},
            {"role": "tool", "tool_call_id": "start_exam_0", "content": result},
        ]
        return self.step()

    def _state_message(self) -> dict:
        """Динамічний стан іспиту: допомагає моделі не губитися між темами."""
        s = self.session
        return {"role": "system", "content": (
            f"[Стан іспиту] Поточна тема: {s.current or 'не обрано'}. "
            f"Залишилось тем у черзі: {len(s.queue)}. "
            f"Повідомлень студента за поточною темою: {s.total_msgs} (змістовних: {s.meaningful}). "
            "Твоє наступне повідомлення має бути питанням або реплікою САМЕ за поточною темою.")}

    def _complete(self):
        """Запит до LLM з повторами при тимчасових помилках API (400 tool_use_failed, 5xx)."""
        last = None
        for _ in range(API_RETRIES):
            try:
                return self.client.chat.completions.create(
                    model=MODEL, messages=self.messages + [self._state_message()],
                    tools=TOOLS, tool_choice="auto", temperature=0.4, max_tokens=1500)
            except APIStatusError as e:
                if e.status_code not in (400, 500, 502, 503):
                    raise
                last = e
        raise last

    def step(self, user_text: str | None = None) -> str:
        """Один хід: (опційно) повідомлення студента -> відповідь бота."""
        self.topic_fetched = False  # не більше одного get_next_topic за хід
        if user_text is not None:
            self.session.note_user_message(user_text)
            self.messages.append({"role": "user", "content": user_text})
            self._log("user", user_text)

        for _ in range(MAX_TOOL_ROUNDS):
            resp = self._complete()
            msg = resp.choices[0].message
            calls = msg.tool_calls or []
            if not calls:
                text = (msg.content or "").strip()
                if not text:  # порожня відповідь — просимо модель відповісти ще раз
                    continue
                self.messages.append({"role": "assistant", "content": text})
                self._log("system", text)
                return text

            self.messages.append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [{"id": c.id, "type": "function", "function": {
                    "name": c.function.name, "arguments": c.function.arguments}}
                    for c in calls]})
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self._run_tool(c.function.name, args)
                self._log("tool_call", f"{c.function.name}({json.dumps(args, ensure_ascii=False)}) -> {result}")
                self.messages.append({"role": "tool", "tool_call_id": c.id, "content": result})
        return "Вибач, сталася внутрішня помилка. Спробуй повторити повідомлення."
