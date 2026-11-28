# ruff: noqa: E501  (a table of sentences and questions: long lines are the data)
"""Build an independent held-out set for the sufficiency gate.

``retrieval-hard@v1`` was used to study the gate and its test split has been read; any rule fitted
or chosen after looking at it can no longer be judged on it. This set is new: other documents, other
facts, other questions, written after the rules under test were fixed, and it is read only at the
end. 24 short documents (English and Russian), 96 answerable questions (48 facts, each asked in its own
and in the other language) and 40 unanswerable ones: 28 *near misses* (the document says something
about the same thing but not what is asked) and 12 about topics the corpus does not touch.

Every record is drafted with the help of a language model and is ``unreviewed``. Nothing here marks
a record as reviewed.

    uv run python eval/scripts/build_gate_set.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "eval" / "corpus" / "gate_v2"
DATASET = ROOT / "eval" / "datasets" / "gate_heldout_v2.json"

Fact = tuple[
    str, str, str, str
]  # sentence, anchor, question in the language, question in the other
DOCS: dict[str, tuple[str, str, list[Fact]]] = {
    "en_cafeteria.md": (
        "en",
        "Cafeteria",
        [
            (
                "The cafeteria opens at 08:00 and closes at 15:30.",
                "opens at 08:00 and closes at 15:30",
                "When does the cafeteria open and close?",
                "Когда открывается и закрывается столовая?",
            ),
            (
                "Vegetarian meals are marked with a green label.",
                "marked with a green label",
                "How are vegetarian meals marked?",
                "Как отмечены вегетарианские блюда?",
            ),
        ],
    ),
    "en_gym.md": (
        "en",
        "Gym",
        [
            (
                "The gym is on the lower ground floor.",
                "on the lower ground floor",
                "Which floor is the gym on?",
                "На каком этаже находится спортзал?",
            ),
            (
                "Gym access requires a signed waiver.",
                "requires a signed waiver",
                "What is needed to use the gym?",
                "Что нужно для посещения спортзала?",
            ),
        ],
    ),
    "en_library.md": (
        "en",
        "Library",
        [
            (
                "The library lends books for 21 days.",
                "lends books for 21 days",
                "For how long can a library book be borrowed?",
                "На какой срок выдаются книги из библиотеки?",
            ),
            (
                "Each member may borrow up to 5 books.",
                "borrow up to 5 books",
                "How many books may one member borrow?",
                "Сколько книг может взять один читатель?",
            ),
        ],
    ),
    "en_shuttle.md": (
        "en",
        "Shuttle bus",
        [
            (
                "The shuttle bus leaves the station every 20 minutes.",
                "every 20 minutes",
                "How often does the shuttle bus leave the station?",
                "Как часто отправляется автобус от станции?",
            ),
            (
                "The last shuttle departs at 21:40.",
                "departs at 21:40",
                "When does the last shuttle depart?",
                "Во сколько уходит последний автобус?",
            ),
        ],
    ),
    "en_badge.md": (
        "en",
        "Badges",
        [
            (
                "Replacement badges cost 12 euros.",
                "cost 12 euros",
                "How much is a replacement badge?",
                "Сколько стоит замена бейджа?",
            ),
            (
                "Badges are printed on Wednesdays.",
                "printed on Wednesdays",
                "On which day are badges printed?",
                "В какой день печатают бейджи?",
            ),
        ],
    ),
    "en_laptop_refresh.md": (
        "en",
        "Laptop refresh",
        [
            (
                "Laptops are replaced every 4 years.",
                "replaced every 4 years",
                "How often are laptops replaced?",
                "Как часто заменяют ноутбуки?",
            ),
            (
                "Developers may choose between two models.",
                "choose between two models",
                "How many laptop models can developers choose from?",
                "Из скольких моделей ноутбука могут выбирать разработчики?",
            ),
        ],
    ),
    "en_conference.md": (
        "en",
        "Conferences",
        [
            (
                "Conference travel needs approval from a director.",
                "needs approval from a director",
                "Who must approve conference travel?",
                "Кто должен согласовать поездку на конференцию?",
            ),
            (
                "Registration fees are paid by the company card.",
                "paid by the company card",
                "How are conference registration fees paid?",
                "Как оплачиваются регистрационные взносы на конференции?",
            ),
        ],
    ),
    "en_phones.md": (
        "en",
        "Company phones",
        [
            (
                "Company phones are replaced after 3 years.",
                "replaced after 3 years",
                "After how many years are company phones replaced?",
                "Через сколько лет заменяют корпоративные телефоны?",
            ),
            (
                "Personal use of company phones is limited to 100 minutes per month.",
                "limited to 100 minutes per month",
                "How many minutes of personal use do company phones allow?",
                "Сколько минут личного использования разрешено на корпоративном телефоне?",
            ),
        ],
    ),
    "en_security_training.md": (
        "en",
        "Security training",
        [
            (
                "Security training is repeated every 12 months.",
                "repeated every 12 months",
                "How often is security training repeated?",
                "Как часто повторяют обучение по безопасности?",
            ),
            (
                "The course takes about 45 minutes.",
                "takes about 45 minutes",
                "How long does the security course take?",
                "Сколько длится курс по безопасности?",
            ),
        ],
    ),
    "en_audit.md": (
        "en",
        "Internal audits",
        [
            (
                "Internal audits take place in March and September.",
                "in March and September",
                "In which months do internal audits take place?",
                "В какие месяцы проходят внутренние аудиты?",
            ),
            (
                "Audit findings must be answered within 21 days.",
                "answered within 21 days",
                "How soon must audit findings be answered?",
                "В какой срок нужно ответить на замечания аудита?",
            ),
        ],
    ),
    "en_holidays.md": (
        "en",
        "Holiday closures",
        [
            (
                "The office is closed on 25 December.",
                "closed on 25 December",
                "Is the office open on 25 December?",
                "Работает ли офис 25 декабря?",
            ),
            (
                "The office is also closed on 1 January.",
                "also closed on 1 January",
                "Is the office open on 1 January?",
                "Работает ли офис 1 января?",
            ),
        ],
    ),
    "en_mentoring.md": (
        "en",
        "Mentoring",
        [
            (
                "Mentoring sessions are held every second Thursday.",
                "every second Thursday",
                "When are mentoring sessions held?",
                "Когда проходят встречи наставничества?",
            ),
            (
                "A mentor may support up to 3 mentees.",
                "support up to 3 mentees",
                "How many mentees may a mentor support?",
                "Сколько подопечных может вести наставник?",
            ),
        ],
    ),
    "ru_cafe.md": (
        "ru",
        "Столовая",
        [
            (
                "Столовая открыта с 8:00 до 15:30.",
                "открыта с 8:00 до 15:30",
                "В какие часы работает столовая?",
                "What are the canteen's opening hours?",
            ),
            (
                "Вегетарианские блюда отмечены зелёной меткой.",
                "отмечены зелёной меткой",
                "Чем отмечены вегетарианские блюда?",
                "What marks the vegetarian dishes in the canteen?",
            ),
        ],
    ),
    "ru_sport.md": (
        "ru",
        "Спортзал",
        [
            (
                "Спортзал находится в цокольном этаже.",
                "в цокольном этаже",
                "Где находится спортзал?",
                "Where is the fitness room located?",
            ),
            (
                "Для посещения спортзала нужно подписать соглашение.",
                "нужно подписать соглашение",
                "Что нужно подписать для посещения спортзала?",
                "What must be signed to use the fitness room?",
            ),
        ],
    ),
    "ru_library.md": (
        "ru",
        "Библиотека",
        [
            (
                "Книги из библиотеки выдаются на 21 день.",
                "выдаются на 21 день",
                "На сколько дней выдают книги из библиотеки?",
                "For how many days are library books lent?",
            ),
            (
                "Один читатель может взять не более 5 книг.",
                "не более 5 книг",
                "Сколько книг может взять один читатель?",
                "What is the limit of books per reader?",
            ),
        ],
    ),
    "ru_shuttle.md": (
        "ru",
        "Служебный автобус",
        [
            (
                "Служебный автобус отходит от станции каждые 20 минут.",
                "каждые 20 минут",
                "Как часто ходит служебный автобус?",
                "How frequent is the company bus?",
            ),
            (
                "Последний автобус уходит в 21:40.",
                "уходит в 21:40",
                "Когда уходит последний служебный автобус?",
                "At what time does the final company bus go?",
            ),
        ],
    ),
    "ru_badge.md": (
        "ru",
        "Пропуска сотрудников",
        [
            (
                "Замена пропуска стоит 600 рублей.",
                "стоит 600 рублей",
                "Сколько стоит замена пропуска?",
                "What does replacing an access card cost?",
            ),
            (
                "Пропуска печатают по средам.",
                "печатают по средам",
                "В какой день печатают пропуска?",
                "On what weekday are access cards printed?",
            ),
        ],
    ),
    "ru_laptop.md": (
        "ru",
        "Замена ноутбуков",
        [
            (
                "Ноутбуки заменяют раз в 4 года.",
                "раз в 4 года",
                "Как часто заменяют ноутбуки сотрудников?",
                "What is the laptop replacement cycle?",
            ),
            (
                "Разработчики могут выбрать одну из двух моделей.",
                "одну из двух моделей",
                "Из скольких моделей выбирают разработчики?",
                "How many models can programmers pick from?",
            ),
        ],
    ),
    "ru_conf.md": (
        "ru",
        "Конференции",
        [
            (
                "Поездка на конференцию требует согласования директора.",
                "требует согласования директора",
                "Кто согласует поездку на конференцию?",
                "Whose consent does a conference trip need?",
            ),
            (
                "Регистрационный взнос оплачивается корпоративной картой.",
                "оплачивается корпоративной картой",
                "Чем оплачивается регистрационный взнос?",
                "What pays the registration fee?",
            ),
        ],
    ),
    "ru_phone.md": (
        "ru",
        "Корпоративные телефоны",
        [
            (
                "Корпоративные телефоны заменяют через 3 года.",
                "заменяют через 3 года",
                "Через сколько лет заменяют корпоративные телефоны?",
                "After what period are work phones replaced?",
            ),
            (
                "Личное использование ограничено 100 минутами в месяц.",
                "ограничено 100 минутами в месяц",
                "Сколько минут в месяц разрешено для личных звонков?",
                "How many private minutes a month are allowed on a work phone?",
            ),
        ],
    ),
    "ru_sec.md": (
        "ru",
        "Обучение по безопасности",
        [
            (
                "Обучение по безопасности повторяется каждые 12 месяцев.",
                "повторяется каждые 12 месяцев",
                "Как часто повторяется обучение по безопасности?",
                "How often must the safety course be retaken?",
            ),
            (
                "Курс занимает около 45 минут.",
                "занимает около 45 минут",
                "Сколько времени занимает курс по безопасности?",
                "How long is the safety course?",
            ),
        ],
    ),
    "ru_audit.md": (
        "ru",
        "Внутренние аудиты",
        [
            (
                "Внутренние аудиты проходят в марте и сентябре.",
                "в марте и сентябре",
                "В какие месяцы проходят внутренние аудиты?",
                "Which months are internal audits held?",
            ),
            (
                "На замечания аудита нужно ответить в течение 21 дня.",
                "в течение 21 дня",
                "В какой срок отвечают на замечания аудита?",
                "Within what time are audit remarks answered?",
            ),
        ],
    ),
    "ru_holidays.md": (
        "ru",
        "Нерабочие дни",
        [
            (
                "Офис закрыт 31 декабря.",
                "закрыт 31 декабря",
                "Работает ли офис 31 декабря?",
                "Does the office work on 31 December?",
            ),
            (
                "Офис закрыт также 8 марта.",
                "закрыт также 8 марта",
                "Работает ли офис 8 марта?",
                "Does the office work on 8 March?",
            ),
        ],
    ),
    "ru_mentor.md": (
        "ru",
        "Наставничество",
        [
            (
                "Встречи наставников проходят каждый второй четверг.",
                "каждый второй четверг",
                "Когда проходят встречи наставников?",
                "When do mentor meetings take place?",
            ),
            (
                "Один наставник может вести не более 3 подопечных.",
                "не более 3 подопечных",
                "Сколько подопечных может вести наставник?",
                "What is the maximum number of mentees per mentor?",
            ),
        ],
    ),
}

NEAR_MISS: list[tuple[str, str, list[str]]] = [
    ("en", "How much does a cafeteria meal cost?", ["meal cost"]),
    ("en", "Who runs the gym?", ["runs the gym"]),
    ("en", "What is the fine for a late library book?", ["fine"]),
    ("en", "Where is the shuttle bus stop at the office?", ["bus stop"]),
    ("en", "Who prints the badges?", ["prints the badges"]),
    ("en", "Which brand are the company laptops?", ["brand"]),
    ("en", "Who books the flights for conference travel?", ["flights"]),
    ("en", "What happens to a company phone after it is replaced?", ["after it is replaced"]),
    ("en", "Who teaches the security training?", ["teaches"]),
    ("en", "Who carries out the internal audits?", ["carries out"]),
    ("en", "Is the office closed on 26 December?", ["26 December"]),
    ("en", "How long does one mentoring session last?", ["session last"]),
    ("en", "Is the cafeteria open on Saturdays?", ["Saturdays"]),
    ("en", "Which weekdays does the shuttle bus run?", ["weekdays"]),
    ("ru", "Сколько стоит обед в столовой?", ["стоит обед"]),
    ("ru", "Кто руководит спортзалом?", ["руководит"]),
    ("ru", "Какой штраф за просрочку библиотечной книги?", ["штраф"]),
    ("ru", "Где остановка служебного автобуса у офиса?", ["остановк"]),
    ("ru", "Кто печатает пропуска?", ["печатает пропуска"]),
    ("ru", "Какой марки выдают ноутбуки?", ["марки"]),
    ("ru", "Кто бронирует билеты на конференцию?", ["билеты"]),
    ("ru", "Что делают с телефоном после замены?", ["после замены"]),
    ("ru", "Кто проводит обучение по безопасности?", ["проводит обучение"]),
    ("ru", "Кто проводит внутренние аудиты?", ["проводит внутренние"]),
    ("ru", "Работает ли офис 2 января?", ["2 января"]),
    ("ru", "Сколько длится одна встреча наставников?", ["одна встреча"]),
    ("ru", "Работает ли столовая по субботам?", ["по субботам"]),
    ("ru", "В какие дни недели ходит служебный автобус?", ["дни недели"]),
]
ABSENT: list[tuple[str, str, list[str]]] = [
    ("en", "What is the company's policy on remote work from abroad?", ["abroad"]),
    ("en", "How many days of sick leave are paid?", ["sick leave"]),
    ("en", "What is the process for requesting a salary review?", ["salary"]),
    ("en", "Which health insurance provider does the company use?", ["insurance"]),
    ("en", "Where can I store my bicycle?", ["bicycle"]),
    ("en", "Who is the data protection officer?", ["protection officer"]),
    ("ru", "Какая политика по работе из другой страны?", ["другой страны"]),
    ("ru", "Сколько дней больничного оплачивается?", ["больничного"]),
    ("ru", "Как запросить пересмотр зарплаты?", ["зарплат"]),
    ("ru", "Какая страховая компания обслуживает сотрудников?", ["страховая"]),
    ("ru", "Где хранить велосипед?", ["велосипед"]),
    ("ru", "Кто отвечает за защиту персональных данных?", ["персональных данных"]),
]

PROVENANCE = {
    "created_by": "llm-assisted",
    "method": "generated from the fact table of eval/scripts/build_gate_set.py after the gate rules "
    "under test were fixed; awaiting human review",
    "created": "2026-11-28",
}
REVIEW = {"status": "unreviewed", "reviewer": None, "date": None, "notes": ""}


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def build() -> tuple[dict[str, str], dict[str, object]]:
    corpus: dict[str, str] = {}
    questions: list[dict[str, object]] = []
    for name, (language, title, facts) in DOCS.items():
        corpus[name] = f"# {title}\n\n" + " ".join(f[0] for f in facts) + "\n"
        for n, (_, anchor, same, cross) in enumerate(facts, start=1):
            other = "ru" if language == "en" else "en"
            for text, lang, tag in ((same, language, "same"), (cross, other, "cross")):
                overlap = len(_words(text) & _words(anchor)) / max(1, len(_words(text)))
                kind = "keyword" if overlap > 0.30 else "paraphrase"
                questions.append(
                    {
                        "id": f"{name.removesuffix('.md')}-{n}-{tag}", "split": "test",
                        "type": kind, "language": lang, "question": text, "answerable": True,
                        "gold": [{"doc": name, "anchor": anchor, "min_support": 1}],
                        "reference_answer": facts[n - 1][0], "reference_facts": [anchor],
                        "absence_terms": [], "provenance": PROVENANCE, "review": REVIEW,
                    }
                )  # fmt: skip
    for kind, group in (("near-miss", NEAR_MISS), ("absent", ABSENT)):
        for n, (lang, text, terms) in enumerate(group, start=1):
            questions.append(
                {
                    "id": f"{kind}-{n}", "split": "test", "type": "no_answer", "language": lang,
                    "question": text, "answerable": False, "gold": [], "reference_answer": None,
                    "reference_facts": [], "absence_terms": terms, "provenance": PROVENANCE,
                    "review": REVIEW,
                }
            )  # fmt: skip
    dataset = {"name": "gate-heldout", "version": 2, "corpus": "eval/corpus/gate_v2", "questions": questions}  # fmt: skip
    return corpus, dataset


def main() -> None:
    corpus, dataset = build()
    CORPUS.mkdir(parents=True, exist_ok=True)
    for name, text in corpus.items():
        (CORPUS / name).write_text(text, encoding="utf-8")
    DATASET.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(corpus)} documents, {len(dataset['questions'])} questions")  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
