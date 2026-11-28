"""Build the harder public corpus and its dataset from one fact table.

The set is meant to be hard for retrieval and for the sufficiency gate, not to be large:

* hard negatives: pairs of documents with the same structure and different numbers (database and
  laptop backups, travel and meal expenses);
* near duplicates and contradictions: two versions of a policy that disagree (``conflicting``);
* multilingual questions: every fact is asked in its own language and in the other one;
* unanswerable questions about topics next to the corpus (``no_answer``), with absence terms that
  the validator checks against the corpus.

Every record is drafted by this script and is ``unreviewed``: it is exploratory material until the
owner reviews it (``rag eval`` and ``review_sheet.py``). Nothing here marks a record as reviewed.

    uv run python eval/scripts/build_hard_dataset.py
"""

# ruff: noqa: E501  (a table of sentences and questions: long lines are the data)
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_revision import record, revise

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "eval" / "corpus" / "hard_v1"
PREVIOUS = ROOT / "eval" / "datasets" / "retrieval_hard_v1.json"
DATASET = ROOT / "eval" / "datasets" / "retrieval_hard_v2.json"
CHANGES = ROOT / "eval" / "datasets" / "CHANGES_v2.json"
VERSION = 2

# file: (language, split, title, [(sentence, anchor, question in the document's language,
#                                  question in the other language)])
Fact = tuple[str, str, str, str]
DOCS: dict[str, tuple[str, str, str, list[Fact]]] = {
    "en_db_backup.md": (
        "en",
        "dev",
        "Database backups",
        [
            (
                "Production database backups run every night at 01:15 UTC.",
                "every night at 01:15 UTC",
                "When do the production database backups run?",
                "В какое время создаются резервные копии производственной базы данных?",
            ),
            (
                "Database backups are encrypted and kept for 30 days.",
                "kept for 30 days",
                "How long are database backups kept?",
                "Как долго хранятся резервные копии базы данных?",
            ),
        ],
    ),
    "en_laptop_backup.md": (
        "en",
        "dev",
        "Laptop backups",
        [
            (
                "Employee laptops are backed up every Friday at 18:00 local time.",
                "every Friday at 18:00 local time",
                "When are employee laptops backed up?",
                "Когда создаются резервные копии ноутбуков сотрудников?",
            ),
            (
                "Laptop backups are kept for 7 days.",
                "kept for 7 days",
                "How long are laptop backups kept?",
                "Как долго хранятся резервные копии ноутбуков?",
            ),
        ],
    ),
    "en_vacation_v1.md": (
        "en",
        "dev",
        "Vacation policy, version 1",
        [
            (
                "In version 1 of the vacation policy, employees receive 20 days of paid vacation per year.",
                "20 days of paid vacation per year",
                "How many vacation days did version 1 of the policy give?",
                "Сколько дней отпуска давала первая версия политики отпусков?",
            ),
            (
                "Unused vacation days under version 1 expire on 31 March.",
                "expire on 31 March",
                "When do unused vacation days expire under version 1?",
                "Когда сгорают неиспользованные дни отпуска по первой версии?",
            ),
        ],
    ),
    "en_vacation_v2.md": (
        "en",
        "dev",
        "Vacation policy, version 2",
        [
            (
                "In version 2 of the vacation policy, employees receive 25 days of paid vacation per year.",
                "25 days of paid vacation per year",
                "How many vacation days does version 2 of the policy give?",
                "Сколько дней отпуска даёт вторая версия политики отпусков?",
            ),
            (
                "Version 2 replaces version 1 from 1 January 2025.",
                "replaces version 1 from 1 January 2025",
                "From when does version 2 of the vacation policy replace version 1?",
                "С какой даты вторая версия политики отпусков заменяет первую?",
            ),
        ],
    ),
    "en_travel.md": (
        "en",
        "dev",
        "Travel expenses",
        [
            (
                "Travel expenses must be submitted within 14 days of the trip.",
                "submitted within 14 days of the trip",
                "How soon after a trip must travel expenses be filed?",
                "Сколько дней после поездки отводится на подачу заявки на возмещение командировочных расходов (Travel expenses)?",
            ),
            (
                "The hotel limit is 150 euros per night.",
                "150 euros per night",
                "What is the nightly hotel limit?",
                "Каков лимит на проживание в гостинице за ночь?",
            ),
        ],
    ),
    "en_meals.md": (
        "en",
        "dev",
        "Meal expenses",
        [
            (
                "Meal expenses must be submitted within 30 days of the purchase.",
                "within 30 days of the purchase",
                "How soon after buying a meal must the expense be filed?",
                "В какой срок после покупки обеда нужно подать расход на питание?",
            ),
            (
                "The meal limit is 40 euros per day.",
                "40 euros per day",
                "What is the daily meal limit?",
                "Каков дневной лимит на питание?",
            ),
        ],
    ),
    "en_vpn.md": (
        "en",
        "test",
        "VPN access",
        [
            (
                "VPN access requires a hardware token issued by the IT desk.",
                "hardware token issued by the IT desk",
                "What do I need to get VPN access?",
                "Что нужно для получения доступа к VPN?",
            ),
            (
                "A lost token must be reported within 2 hours.",
                "reported within 2 hours",
                "How quickly must a lost VPN token be reported?",
                "Как быстро нужно сообщить об утере токена VPN?",
            ),
        ],
    ),
    "en_password.md": (
        "en",
        "test",
        "Password rules",
        [
            (
                "Passwords must contain at least 14 characters.",
                "at least 14 characters",
                "How long must a password be?",
                "Какой минимальной длины должен быть пароль?",
            ),
            (
                "Passwords expire every 180 days and the last 10 passwords cannot be reused.",
                "expire every 180 days",
                "How often do passwords have to be changed?",
                "Как часто нужно менять пароль?",
            ),
        ],
    ),
    "en_incident.md": (
        "en",
        "test",
        "Incident response",
        [
            (
                "Severity 1 incidents must be acknowledged within 5 minutes.",
                "acknowledged within 5 minutes",
                "How fast must a severity 1 incident be acknowledged?",
                "За какое время нужно подтвердить инцидент первой степени серьёзности?",
            ),
            (
                "The incident commander is the on-call lead.",
                "incident commander is the on-call lead",
                "Who acts as incident commander?",
                "Кто выступает руководителем инцидента?",
            ),
        ],
    ),
    "en_onboarding.md": (
        "en",
        "test",
        "Onboarding",
        [
            (
                "New employees receive a laptop on their first day.",
                "receive a laptop on their first day",
                "When does a new employee get a laptop?",
                "Когда новый сотрудник получает ноутбук?",
            ),
            (
                "The buddy program lasts 90 days.",
                "buddy program lasts 90 days",
                "How long does the buddy program run?",
                "Сколько длится программа наставничества?",
            ),
        ],
    ),
    "en_release.md": (
        "en",
        "test",
        "Release schedule",
        [
            (
                "Production releases happen every Tuesday at 14:00 UTC.",
                "every Tuesday at 14:00 UTC",
                "On which day and time are production releases made?",
                "В какой день и в какое время по UTC выходят релизы в продакшн?",
            ),
            (
                "A release freeze applies in the last week of December.",
                "release freeze applies in the last week of December",
                "When is the release freeze?",
                "Когда действует заморозка релизов?",
            ),
        ],
    ),
    "en_parking.md": (
        "en",
        "test",
        "Parking",
        [
            (
                "Visitors park in zone C for up to 4 hours.",
                "zone C for up to 4 hours",
                "Where and for how long may visitors park?",
                "Где и на какое время могут парковаться гости?",
            ),
            (
                "Parking passes are issued by the facilities team.",
                "issued by the facilities team",
                "Who issues parking passes?",
                "Кто выдаёт парковочные пропуска?",
            ),
        ],
    ),
    "ru_mail_backup.md": (
        "ru",
        "dev",
        "Резервное копирование почты",
        [
            (
                "Резервные копии почтового сервера создаются каждую ночь в 03:20 по московскому времени.",
                "каждую ночь в 03:20 по московскому времени",
                "Когда создаются резервные копии почтового сервера?",
                "When are the mail server backups created?",
            ),
            (
                "Копии почтового сервера хранятся 21 день.",
                "хранятся 21 день",
                "Сколько хранятся копии почтового сервера?",
                "How long are the mail server backups kept?",
            ),
        ],
    ),
    "ru_office_hours.md": (
        "ru",
        "dev",
        "Режим работы офиса",
        [
            (
                "Офис открыт с понедельника по пятницу с 9:00 до 18:00.",
                "с 9:00 до 18:00",
                "В какие часы работает офис?",
                "During which hours is the office open?",
            ),
            (
                "В субботу и воскресенье офис закрыт.",
                "В субботу и воскресенье офис закрыт",
                "Работает ли офис в выходные?",
                "Is the office open at the weekend?",
            ),
        ],
    ),
    "ru_travel.md": (
        "ru",
        "dev",
        "Командировки",
        [
            (
                "Отчёт о командировке нужно сдать в течение 10 рабочих дней.",
                "в течение 10 рабочих дней",
                "В какой срок нужно сдать отчёт о командировке?",
                "Within how many working days must a business trip report be submitted?",
            ),
            (
                "Суточные составляют 1200 рублей в день.",
                "1200 рублей в день",
                "Каков размер суточных?",
                "How many roubles of per diem are paid for each day of a business trip?",
            ),
        ],
    ),
    "ru_sick.md": (
        "ru",
        "dev",
        "Больничные листы",
        [
            (
                "Больничный лист нужно передать в отдел кадров в течение 3 дней после выхода на работу.",
                "в течение 3 дней после выхода на работу",
                "Когда нужно передать больничный лист в отдел кадров?",
                "When must a sick note be handed to HR?",
            ),
            (
                "Первые 3 дня болезни оплачивает работодатель.",
                "оплачивает работодатель",
                "Кто оплачивает первые дни болезни?",
                "Who pays for the first days of illness?",
            ),
        ],
    ),
    "ru_remote.md": (
        "ru",
        "dev",
        "Удалённый доступ",
        [
            (
                "Для удалённого доступа используется двухфакторная аутентификация.",
                "двухфакторная аутентификация",
                "Какая аутентификация нужна для удалённого доступа?",
                "What authentication does remote access need?",
            ),
            (
                "Сессия удалённого доступа завершается после 30 минут бездействия.",
                "после 30 минут бездействия",
                "Когда завершается сессия удалённого доступа?",
                "When does a remote access session end?",
            ),
        ],
    ),
    "ru_training.md": (
        "ru",
        "dev",
        "Обучение сотрудников",
        [
            (
                "Бюджет на обучение составляет 80000 рублей в год на сотрудника.",
                "80000 рублей в год на сотрудника",
                "Какой бюджет на обучение выделяется сотруднику?",
                "What training budget does an employee get?",
            ),
            (
                "Заявку на курс нужно согласовать с руководителем за 14 дней.",
                "согласовать с руководителем за 14 дней",
                "За сколько дней нужно согласовать заявку на курс?",
                "How many days ahead must a course request be approved?",
            ),
        ],
    ),
    "ru_purchase.md": (
        "ru",
        "test",
        "Закупки",
        [
            (
                "Закупки свыше 100000 рублей согласовывает финансовый директор.",
                "свыше 100000 рублей согласовывает финансовый директор",
                "Кто согласовывает крупные закупки?",
                "Who approves large purchases?",
            ),
            (
                "Три коммерческих предложения требуются для закупок свыше 500000 рублей.",
                "свыше 500000 рублей",
                "Для каких закупок нужны три коммерческих предложения?",
                "Which purchases need three quotations?",
            ),
        ],
    ),
    "ru_pass.md": (
        "ru",
        "test",
        "Пропуска",
        [
            (
                "Постоянный пропуск выдаёт служба безопасности в течение 2 рабочих дней.",
                "в течение 2 рабочих дней",
                "За сколько дней выдают постоянный пропуск?",
                "How long does it take to get a permanent pass?",
            ),
            (
                "Утерянный пропуск нужно заблокировать в течение 1 часа.",
                "в течение 1 часа",
                "В какой срок блокируют утерянный пропуск?",
                "How soon must a lost pass be blocked?",
            ),
        ],
    ),
    "ru_kitchen.md": (
        "ru",
        "test",
        "Кухня и переговорные",
        [
            (
                "Переговорные комнаты бронируются через календарь не более чем на 2 часа.",
                "не более чем на 2 часа",
                "На какое время можно забронировать переговорную?",
                "For how long can a meeting room be booked?",
            ),
            (
                "Кухня находится на втором этаже рядом с лифтом.",
                "на втором этаже рядом с лифтом",
                "Где находится кухня?",
                "Where is the kitchen?",
            ),
        ],
    ),
    "ru_release.md": (
        "ru",
        "test",
        "Релизы",
        [
            (
                "Релизы выходят каждый четверг в 16:00 по московскому времени.",
                "каждый четверг в 16:00 по московскому времени",
                "Когда выходят релизы?",
                "On which day and at what Moscow time do releases go out?",
            ),
            (
                "Откат релиза допускается в течение 24 часов.",
                "в течение 24 часов",
                "В какой срок допускается откат релиза?",
                "How long is a release rollback allowed?",
            ),
        ],
    ),
    "ru_data.md": (
        "ru",
        "test",
        "Хранение данных",
        [
            (
                "Журналы доступа хранятся 12 месяцев.",
                "хранятся 12 месяцев",
                "Как долго хранятся журналы доступа?",
                "How long are access logs kept?",
            ),
            (
                "Персональные данные удаляются по запросу в течение 30 дней.",
                "в течение 30 дней",
                "В какой срок удаляют персональные данные по запросу?",
                "How fast is personal data deleted on request?",
            ),
        ],
    ),
    "ru_it.md": (
        "ru",
        "test",
        "Заявки в ИТ",
        [
            (
                "Заявки в ИТ-поддержку принимаются через портал круглосуточно.",
                "через портал круглосуточно",
                "Как подать заявку в ИТ-поддержку?",
                "How do I submit an IT support request?",
            ),
            (
                "Критичные заявки обрабатываются в течение 15 минут.",
                "в течение 15 минут",
                "За какое время обрабатывают критичные заявки?",
                "How fast are critical IT support requests handled?",
            ),
        ],
    ),
    "en_remote_2023.md": (
        "en",
        "test",
        "Remote work policy 2023",
        [
            (
                "Under the 2023 remote work policy, employees may work remotely 2 days per week.",
                "work remotely 2 days per week",
                "How many days per week could employees work remotely in 2023?",
                "Сколько дней в неделю сотрудники могли работать удалённо в 2023 году?",
            ),
        ],
    ),
    "en_remote_2024.md": (
        "en",
        "test",
        "Remote work policy 2024",
        [
            (
                "Under the 2024 remote work policy, employees may work remotely up to 3 days per week.",
                "work remotely up to 3 days per week",
                "How many days per week may employees work remotely in 2024?",
                "Сколько дней в неделю сотрудники могут работать удалённо в 2024 году?",
            ),
            (
                "The 2024 policy replaces the 2023 policy.",
                "replaces the 2023 policy",
                "Which remote work policy replaces the 2023 one?",
                "Какая политика удалённой работы заменяет политику 2023 года?",
            ),
        ],
    ),
}

# (type, split, question, language, gold [(doc, anchor)], reference answer)
GROUP: list[tuple[str, str, str, str, list[tuple[str, str]], str]] = [
    (
        "multi_doc",
        "dev",
        "Compare how long database backups and laptop backups are kept.",
        "en",
        [("en_db_backup.md", "kept for 30 days"), ("en_laptop_backup.md", "kept for 7 days")],
        "Database backups are kept for 30 days and laptop backups for 7 days.",
    ),
    (
        "multi_doc",
        "dev",
        "Какие сроки подачи расходов установлены для командировок и для питания?",
        "ru",
        [
            ("en_travel.md", "submitted within 14 days of the trip"),
            ("en_meals.md", "within 30 days of the purchase"),
        ],
        "Travel expenses within 14 days, meal expenses within 30 days.",
    ),
    (
        "multi_doc",
        "dev",
        "Сколько хранятся копии почтового сервера и резервные копии базы данных?",
        "ru",
        [("ru_mail_backup.md", "хранятся 21 день"), ("en_db_backup.md", "kept for 30 days")],
        "Копии почтового сервера хранятся 21 день, резервные копии базы данных - 30 дней.",
    ),
    (
        "multi_doc",
        "test",
        "What are the password length rule and the VPN token rule?",
        "en",
        [
            ("en_password.md", "at least 14 characters"),
            ("en_vpn.md", "hardware token issued by the IT desk"),
        ],
        "Passwords need at least 14 characters; VPN access needs a hardware token.",
    ),
    (
        "multi_doc",
        "test",
        "В какой срок допускается откат релиза и за какое время обрабатываются критичные заявки?",
        "ru",
        [("ru_release.md", "в течение 24 часов"), ("ru_it.md", "в течение 15 минут")],
        "Откат релиза допускается в течение 24 часов; критичные заявки обрабатываются в течение 15 минут.",
    ),
    (
        "conflicting",
        "dev",
        "How many vacation days do employees get per year?",
        "en",
        [
            ("en_vacation_v1.md", "20 days of paid vacation per year"),
            ("en_vacation_v2.md", "25 days of paid vacation per year"),
        ],
        "Version 1 gave 20 days; version 2 gives 25 days and replaces version 1 from 1 January 2025.",
    ),
    (
        "conflicting",
        "dev",
        "Сколько дней оплачиваемого отпуска положено в год?",
        "ru",
        [
            ("en_vacation_v1.md", "20 days of paid vacation per year"),
            ("en_vacation_v2.md", "25 days of paid vacation per year"),
        ],
        "Version 1 gave 20 days; version 2 gives 25 days and replaces version 1 from 1 January 2025.",
    ),
    (
        "conflicting",
        "test",
        "How many days per week may employees work remotely?",
        "en",
        [
            ("en_remote_2023.md", "work remotely 2 days per week"),
            ("en_remote_2024.md", "work remotely up to 3 days per week"),
        ],
        "2 days under the 2023 policy; up to 3 days under the 2024 policy, which replaces it.",
    ),
    (
        "conflicting",
        "test",
        "Сколько дней в неделю разрешена удалённая работа?",
        "ru",
        [
            ("en_remote_2023.md", "work remotely 2 days per week"),
            ("en_remote_2024.md", "work remotely up to 3 days per week"),
        ],
        "2 days under the 2023 policy; up to 3 days under the 2024 policy, which replaces it.",
    ),
]

# Each question is asked in one language; its absence terms cover both languages, because the
# corpus has documents in both and an answer in the other language would still be an answer.
UNANSWERABLE: list[tuple[str, str, list[str]]] = [
    (
        "en",
        "What is the dental insurance coverage?",
        ["dental", "insurance", "стоматолог", "страхов"],
    ),
    ("en", "What is the policy on working from abroad?", ["abroad", "за границ", "за рубеж"]),
    ("en", "How much is the monthly gym allowance?", ["gym", "спортзал", "фитнес"]),
    ("en", "What is the dress code on Fridays?", ["dress code", "дресс-код", "одежд"]),
    ("en", "How many days of parental leave are granted?", ["parental", "декрет", "родительск"]),
    (
        "en",
        "Who is the data protection officer?",
        ["protection officer", "защите данных", "уполномоченн"],
    ),
    ("en", "When is the annual salary review?", ["salary", "зарплат", "оклад"]),
    ("en", "How are stock options vested?", ["stock option", "опцион", "акци"]),
    (
        "en",
        "What is the notice period for resignation?",
        ["resignation", "увольнени", "заявление об уходе"],
    ),
    ("en", "Where can I store my bicycle?", ["bicycle", "велосипед", "велопарков"]),
    ("en", "How do I request a standing desk?", ["standing desk", "стол с регулировк"]),
    ("en", "What is the Wi-Fi password for guests?", ["wi-fi", "гостев"]),
    (
        "en",
        "Which languages does the language course offer?",
        ["language course", "языковые курсы", "иностранн"],
    ),
    ("en", "How many monitors does each employee get?", ["monitor", "монитор"]),
    ("en", "What is the retention period for payroll records?", ["payroll", "расчётн", "зарплатн"]),
    ("ru", "Какая страховка предусмотрена для сотрудников?", ["страховк", "insurance", "medical"]),
    (
        "ru",
        "Сколько дней отпуска даётся при рождении ребёнка?",
        ["рождении ребёнка", "childbirth", "birth", "maternity"],
    ),
    ("ru", "Какой размер премии по итогам года?", ["премии", "bonus"]),
    ("ru", "Как оформить увольнение?", ["увольнени", "resign", "dismissal", "termination"]),
    ("ru", "Где находится стоянка для велосипедов?", ["велосипед", "bicycle", "bike"]),
    ("ru", "Какой пароль от гостевого Wi-Fi?", ["wi-fi", "гостев", "guest network"]),
    ("ru", "Сколько стоит обед в столовой?", ["столов", "canteen", "cafeteria", "lunch"]),
    ("ru", "Как получить доступ к бухгалтерской системе?", ["бухгалтерск", "accounting"]),
    (
        "ru",
        "Какова процедура аттестации сотрудников?",
        ["аттестаци", "appraisal", "performance review"],
    ),
    ("ru", "Кто отвечает за охрану труда?", ["охран", "occupational safety", "safety officer"]),
    ("ru", "Сколько длится испытательный срок?", ["испытательн", "probation"]),
    ("ru", "Какая скидка на корпоративный фитнес?", ["фитнес", "fitness", "gym"]),
    ("ru", "Как заказать канцелярские товары?", ["канцеляр", "stationery", "office supplies"]),
    ("ru", "Как сообщить о конфликте интересов?", ["конфликт", "conflict of interest"]),
    ("ru", "Какая политика по работе из дома?", ["из дома", "work from home", "working from home"]),
]

PROVENANCE = {
    "created_by": "llm-assisted",
    "method": "generated from the fact table of eval/scripts/build_hard_dataset.py; "
    "awaiting human review",
    "created": "2026-11-28",
}
REVIEW = {"status": "unreviewed", "reviewer": None, "date": None, "notes": ""}


# Why each question differs from version 1. ``revise`` refuses a change that is not listed here and
# a line here for a question that did not change, so this table is the history.
REASONS: dict[str, tuple[str, str]] = {
    "multi_doc-3": (
        "gold",
        "the third gold quote (office hours) had nothing to do with the question; it was added only to satisfy 'two documents'. The question now needs two real documents",
    ),
    "multi_doc-5": (
        "gold",
        "the third gold quote (critical requests) was unrelated to the release question; the question now asks about rollback and critical requests, one document each",
    ),
    "ru_office_hours-2-same": (
        "anchor",
        "the quote 'офис закрыт' was too short to identify the evidence",
    ),
    "ru_office_hours-2-cross": (
        "anchor",
        "the quote 'офис закрыт' was too short to identify the evidence",
    ),
    "ru_training-2-same": (
        "anchor",
        "the quote 'за 14 дней' was too short to identify the evidence; the question type is recomputed from the longer quote",
    ),
    "ru_training-2-cross": (
        "anchor",
        "the quote 'за 14 дней' was too short to identify the evidence; the question type is recomputed from the longer quote",
    ),
    "en_travel-1-cross": (
        "ambiguity",
        "'отчёт о командировочных расходах' also matched the trip report of ru_travel (10 working days); it now asks for the filing of expenses and names the policy's own term, Travel expenses",
    ),
    "conflicting-7": (
        "completeness",
        "the expected answer left out that version 2 replaces version 1 (the English twin conflicting-6 has it), so the current rule was missing",
    ),
    "conflicting-9": (
        "completeness",
        "the expected answer left out that the 2024 policy replaces the 2023 one (the English twin conflicting-8 has it)",
    ),
    "ru_travel-1-cross": (
        "ambiguity",
        "'trip report' also matched the expense filing of en_travel (14 days); it now says working days and a business trip report",
    ),
    "ru_travel-2-cross": (
        "ambiguity",
        "'daily travel allowance' also matched the meal limit per day of en_meals; it now asks for the per diem in roubles",
    ),
    "en_release-1-cross": (
        "ambiguity",
        "the Russian question also matched the Thursday schedule of ru_release; it now names UTC, which only en_release uses",
    ),
    "ru_release-1-cross": (
        "ambiguity",
        "the English question also matched the Tuesday schedule of en_release; it now names Moscow time, which only ru_release uses",
    ),
    "ru_it-2-cross": (
        "ambiguity",
        "'critical requests' could be taken for the severity 1 incidents of en_incident; it now says IT support requests",
    ),
}
for _n in range(1, 31):  # every unanswerable question gets terms in both languages
    REASONS[f"no-answer-{_n}"] = (
        "absence",
        "absence terms now cover both languages, because the corpus has documents in both",
    )


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def build() -> tuple[dict[str, str], dict[str, object]]:
    corpus: dict[str, str] = {}
    questions: list[dict[str, object]] = []
    for name, (language, split, title, facts) in DOCS.items():
        corpus[name] = f"# {title}\n\n" + " ".join(f[0] for f in facts) + "\n"
        for n, (_, anchor, same, cross) in enumerate(facts, start=1):
            other = "ru" if language == "en" else "en"
            for asked, lang in ((same, language), (cross, other)):
                overlap = len(_words(asked) & _words(anchor)) / max(1, len(_words(asked)))
                kind = "keyword" if overlap > 0.30 else "paraphrase"
                tag = "same" if lang == language else "cross"
                questions.append(
                    {
                        "id": f"{name.removesuffix('.md')}-{n}-{tag}",
                        "split": split, "type": kind, "language": lang, "question": asked,
                        "answerable": True,
                        "gold": [{"doc": name, "anchor": anchor, "min_support": 1}],
                        "reference_answer": facts[n - 1][0], "reference_facts": [anchor],
                        "absence_terms": [], "provenance": PROVENANCE, "review": REVIEW,
                    }
                )  # fmt: skip
    for n, (kind, split, text, lang, gold, answer) in enumerate(GROUP, start=1):
        questions.append(
            {
                "id": f"{kind}-{n}", "split": split, "type": kind, "language": lang,
                "question": text, "answerable": True,
                "gold": [{"doc": d, "anchor": a, "min_support": 1} for d, a in gold],
                "reference_answer": answer, "reference_facts": [a for _, a in gold],
                "absence_terms": [], "provenance": PROVENANCE, "review": REVIEW,
            }
        )  # fmt: skip
    for n, (lang, text, terms) in enumerate(UNANSWERABLE, start=1):
        questions.append(
            {
                "id": f"no-answer-{n}", "split": "dev" if n % 2 else "test", "type": "no_answer",
                "language": lang, "question": text, "answerable": False, "gold": [],
                "reference_answer": None, "reference_facts": [], "absence_terms": terms,
                "provenance": PROVENANCE, "review": REVIEW,
            }
        )  # fmt: skip
    dataset = {"name": "retrieval-hard", "version": VERSION, "corpus": "eval/corpus/hard_v1",
               "questions": questions}  # fmt: skip
    return corpus, dataset


def check(dataset: dict[str, object], corpus: dict[str, str]) -> None:
    """Refuse to write a dataset the audit finds wrong on its face (the root-cause guard).

    The first version carried two multi-document questions with a gold quote that had nothing to do
    with them. The generator now runs the audit's structural and evidence checks on its own output.
    """
    from rag_engine.evaluation.dataset import parse_dataset, validate_dataset
    from rag_engine.evaluation.question_audit import Verdict, audit_source
    from rag_engine.evaluation.review_kit import SourceDataset

    parsed = parse_dataset(dataset)
    report = validate_dataset(parsed, corpus)
    if not report.ok:
        raise SystemExit("the dataset does not validate:\n" + "\n".join(report.errors))
    audits = audit_source(SourceDataset("retrieval_hard_v2", dataset, parsed, corpus))
    wrong = [a for a in audits if a.verdict is Verdict.INCORRECT]
    if wrong:
        raise SystemExit(
            "the audit finds questions wrong on their face:\n"
            + "\n".join(f"{a.key}: {a.hint}" for a in wrong)
        )


def produce() -> tuple[dict[str, str], dict[str, object], list[dict[str, object]], int]:
    """The corpus, the new dataset version with its provenance, its changes and the previous version.

    Nothing is written; the guard runs here, so a test sees exactly what ``main`` would write.
    """
    corpus, dataset = build()
    check(dataset, corpus)
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    changes = revise(previous, dataset, REASONS)
    return corpus, dataset, changes, int(previous["version"])


def main() -> None:
    corpus, dataset, changes, previous_version = produce()
    CORPUS.mkdir(parents=True, exist_ok=True)
    for name, text in corpus.items():
        (CORPUS / name).write_text(text, encoding="utf-8")
    DATASET.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    record(CHANGES, "retrieval-hard", previous_version, VERSION, "2026-11-28", changes)
    print(f"{len(corpus)} documents, {len(dataset['questions'])} questions")  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
