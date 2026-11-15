import json
import random
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
DATA = BASE / "data"
RESULTS = BASE / "results"
NO_MORE_TOPICS = "NO_MORE_TOPICS"


def _load(name: str):
    return json.loads((DATA / name).read_text(encoding="utf-8"))

def now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()

def _safe(email: str) -> str:
    return "".join(c if c.isalnum() or c in "@._-" else "_" for c in email)

def start_exam(email: str, name: str) -> list[str]:
    """Перевіряє студента, фіксує початок іспиту, повертає 2-3 випадкові теми.
    Кидає ValueError, якщо студента не знайдено.
    """
    email_n, name_n = email.strip().lower(), " ".join(name.split()).lower()
    student = next((s for s in _load("students.json")
                    if s["email"].lower() == email_n), None)
    if student is None:
        raise ValueError(f"Студента з email '{email}' не знайдено в базі.")
    if student["name"].lower() != name_n:
        raise ValueError("Ім'я не збігається з email у базі. Уточніть дані.")

    topics = random.sample(_load("topics.json"), k=random.choice([2, 3]))
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{_safe(email_n)}.json"
    path.write_text(json.dumps(
        {"email": email_n, "name": student["name"], "started_at": now(),
         "topics": topics, "status": "started"}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return topics

def end_exam(email: str, score: float, history: list[dict]) -> None:
    """Записує результат іспиту (score 0..10) та повну історію чату."""
    score = max(0.0, min(10.0, float(score)))
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{_safe(email.strip().lower())}.json"
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"email": email}
    record.update({"status": "finished", "finished_at": now(),
                   "score": score, "history": history})
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

class ExamSession:
    """Стан одного іспиту (окремий для кожного користувача Streamlit)."""

    def __init__(self):
        self.email: str | None = None
        self.queue: list[str] = []
        self.current: str | None = None
        self.finished = False
        self.exhausted = False       # get_next_topic вже повернув NO_MORE_TOPICS
        self.meaningful = 0          # змістовні відповіді на поточну тему
        self.total_msgs = 0          # усі повідомлення студента на поточну тему

    def note_user_message(self, text: str) -> None:
        """Рахує повідомлення студента по поточній темі ('.' чи 'ys' не вважаємо відповіддю)."""
        self.total_msgs += 1
        if sum(ch.isalpha() for ch in text) >= 3:
            self.meaningful += 1

    def start_exam(self, email: str, name: str) -> list[str]:
        topics = start_exam(email, name)
        self.email, self.queue = email.strip().lower(), list(topics)
        return topics

    def get_next_topic(self) -> str:
        """Повертає наступну тему зі списку або NO_MORE_TOPICS."""
        if not self.email:
            raise ValueError("Іспит ще не розпочато: спочатку викличте start_exam.")
        if self.current is not None and self.meaningful < 1 and self.total_msgs < 3:
            raise ValueError(
                "Студент ще не дав змістовної відповіді на поточну тему. НЕ переходь далі: "
                "попроси відповісти або запитай, чи він не знає відповіді.")
        self.meaningful = self.total_msgs = 0
        if not self.queue:
            self.current, self.exhausted = None, True
            return NO_MORE_TOPICS
        self.current = self.queue.pop(0)
        return self.current
