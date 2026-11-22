"""Writes the ``corpus_mini`` fixture documents (Russian and English, several formats).

The texts are produced from fixed phrase banks with a small deterministic generator, so the
files are reproducible: ``python tests/fixtures/generate_corpus_mini.py`` rewrites them in
``tests/fixtures/corpus_mini``. The corpus is original content invented for the tests.
"""

from __future__ import annotations

import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from tests.pdf_builder import build_pdf

OUT = Path(__file__).parent / "corpus_mini"


class Lcg:
    """A tiny deterministic generator, independent of the interpreter version."""

    def __init__(self, seed: int) -> None:
        self.state = seed

    def next(self, bound: int) -> int:
        self.state = (self.state * 1103515245 + 12345) % 2**31
        return (self.state >> 8) % bound

    def pick(self, items: Sequence[str]) -> str:
        return items[self.next(len(items))]


EN_CLAUSES = [
    "The nightly backup job copies every changed file to the archive volume",
    "The release manager checks the build report before tagging a version",
    "The scheduler starts the export when the queue is empty",
    "The storage layer keeps two generations of every snapshot",
    "The on-call engineer reviews the alert history at the start of a shift",
    "The configuration loader reads the file once and validates every section",
    "The audit log records who changed a setting and when",
    "The import step skips files that have not changed since the last run",
    "The retention policy removes old snapshots after the grace period",
    "The monitoring agent reports free disk space every few minutes",
]
EN_TAILS = [
    "and writes a short summary to the log",
    "unless the previous run is still active",
    "so that nothing is lost after a restart",
    "before any other step is allowed to begin",
    "and stops with a clear message when a check fails",
    "which keeps the process predictable for the whole team",
    "while the rest of the system keeps working",
    "and the result is stored together with its checksum",
]
RU_CLAUSES = [
    "Ночное резервное копирование переносит все изменённые файлы на архивный том",
    "Ответственный за выпуск проверяет отчёт о сборке перед присвоением версии",
    "Планировщик запускает выгрузку, когда очередь пуста",
    "Слой хранения держит два поколения каждого снимка",
    "Дежурный инженер просматривает историю оповещений в начале смены",
    "Загрузчик настроек читает файл один раз и проверяет каждый раздел",
    "Журнал аудита фиксирует, кто и когда изменил параметр",
    "Шаг импорта пропускает файлы, которые не менялись с прошлого запуска",
    "Политика хранения удаляет старые снимки по истечении отсрочки",
    "Агент мониторинга сообщает о свободном месте на диске каждые несколько минут",
]
RU_TAILS = [
    "и записывает краткий итог в журнал",
    "если предыдущий запуск уже завершён",
    "чтобы ничего не пропало после перезапуска",
    "прежде чем начнётся любой другой шаг",
    "и останавливается с понятным сообщением при сбое проверки",
    "что делает процесс предсказуемым для всей команды",
    "пока остальная система продолжает работать",
    "а результат хранится вместе с контрольной суммой",
]


def sentences(rnd: Lcg, lang: str, count: int, tag: int) -> list[str]:
    clauses, tails = (EN_CLAUSES, EN_TAILS) if lang == "en" else (RU_CLAUSES, RU_TAILS)
    word = "step" if lang == "en" else "шаг"
    return [f"{rnd.pick(clauses)} {rnd.pick(tails)} ({word} {tag}.{i + 1})." for i in range(count)]


def paragraphs(rnd: Lcg, lang: str, count: int, size: int = 4, tag: int = 1) -> str:
    return "\n\n".join(" ".join(sentences(rnd, lang, size, tag * 10 + n)) for n in range(count))


def md_section(rnd: Lcg, lang: str, level: int, title: str, count: int, tag: int) -> str:
    return f"{'#' * level} {title}\n\n{paragraphs(rnd, lang, count, tag=tag)}"


def write(name: str, text: str | bytes) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text.rstrip() + "\n", encoding="utf-8")


def html_page(lang: str, title: str, body: str) -> str:
    return (
        f'<!DOCTYPE html>\n<html lang="{lang}"><head><meta charset="utf-8">'
        f"<title>{title}</title></head><body>\n"
        '<nav><a href="/">Home</a> <a href="/docs">Docs</a></nav>\n'
        f"<main>\n{body}\n</main>\n<footer>Copyright notice and links</footer>\n</body></html>"
    )


def english(rnd: Lcg) -> None:
    table = "| Setting | Default | Meaning |\n|---|---|---|\n" + "\n".join(
        f"| option_{i} | {i * 5} | limit number {i} in seconds |" for i in range(1, 25)
    )
    code = "\n".join(
        [
            "def run_backup(source, target, keep=7):",
            "    snapshot = take_snapshot(source)",
            "    copy_files(snapshot, target)",
            "    prune(target, keep=keep)",
            "    return verify(target)",
        ]
        * 3
    )
    restore = "- Stop the writers.\n- Copy the snapshot back.\n- Start the writers."
    write(
        "en_backup_guide.md",
        "\n\n".join(
            [
                "# Backup guide",
                paragraphs(rnd, "en", 2, tag=1),
                md_section(rnd, "en", 2, "Preparing the volume", 2, 2),
                md_section(rnd, "en", 3, "Checking free space", 3, 3),
                "## Running a backup\n\n" + paragraphs(rnd, "en", 1, tag=4),
                f"```python\n{code}\n```",
                "## Settings\n\n" + table,
                f"## Restore\n\n{restore}\n\n" + paragraphs(rnd, "en", 2, tag=5),
            ]
        ),
    )
    write(
        "en_release_process.md",
        "\n\n".join(
            [
                "# Release process",
                "Short overview.",
                "## Plan\n\nPick the date.",
                "## Build\n\n" + paragraphs(rnd, "en", 3, tag=6),
                "### Checks\n\nRun all tests.",
                "### Sign-off\n\nAsk the owner.",
                "## Publish\n\n" + paragraphs(rnd, "en", 2, tag=7),
            ]
        ),
    )
    write("en_faq.txt", "\n\n".join(paragraphs(rnd, "en", 1, 3, tag=n) for n in range(8, 20)))
    write(
        "en_changelog.md",
        "# Changelog\n\n"
        + "\n\n".join(
            f"## Version 1.{i}\n\n- Fixed issue number {i * 7}.\n- Improved the import step."
            for i in range(9)
        ),
    )
    items = paragraphs(rnd, "en", 4, tag=30).split("\n\n")
    rows = "".join(
        f"<tr><td>component {i}</td><td>running &amp; healthy</td></tr>" for i in range(6)
    )
    write(
        "en_status_page.html",
        html_page(
            "en",
            "Service status",
            "<h1>Service status</h1>\n"
            + "\n".join(f"<p>{p}</p>" for p in items)
            + "\n<h2>Components</h2>\n<table><tr><th>Name</th><th>State</th></tr>"
            + rows
            + "</table>",
        ),
    )
    write("en_single_paragraph.txt", " ".join(sentences(rnd, "en", 60, 40)))
    write("en_report.pdf", pdf_report(rnd, "en", "Backup report", 100))
    write(
        "en_config_reference.md",
        "# Configuration reference\n\n"
        + table
        + "\n\n## Notes\n\n"
        + paragraphs(rnd, "en", 2, tag=41),
    )


def russian(rnd: Lcg) -> None:
    steps = "1. Остановите запись.\n2. Запустите задачу.\n3. Проверьте итог."
    write(
        "ru_instruction.md",
        "\n\n".join(
            [
                "# Инструкция по резервному копированию",
                paragraphs(rnd, "ru", 2, tag=1),
                md_section(rnd, "ru", 2, "Подготовка тома", 2, 2),
                f"## Запуск\n\n{steps}\n\n" + paragraphs(rnd, "ru", 2, tag=3),
                md_section(rnd, "ru", 2, "Восстановление", 3, 4),
            ]
        ),
    )
    write("ru_policy.txt", "\n\n".join(paragraphs(rnd, "ru", 1, 4, tag=n) for n in range(50, 60)))
    write(
        "ru_faq.md",
        "# Частые вопросы\n\n"
        + "\n\n".join(
            f"## Вопрос {i}\n\n{paragraphs(rnd, 'ru', 1, 2, tag=60 + i)}" for i in range(1, 8)
        ),
    )
    items = paragraphs(rnd, "ru", 4, tag=70).split("\n\n")
    notes = "".join(f"<li>Пункт {i} &mdash; без замечаний</li>" for i in range(1, 6))
    write(
        "ru_page.html",
        html_page(
            "ru",
            "Состояние сервиса",
            "<h1>Состояние сервиса</h1>\n"
            + "\n".join(f"<p>{p}</p>" for p in items)
            + f"\n<h2>Заметки</h2>\n<ul>{notes}</ul>",
        ),
    )
    repeated = "Проверка завершена без замечаний. Журнал сохранён в архив."
    write(
        "ru_repeated_notes.txt",
        "\n\n".join([repeated, paragraphs(rnd, "ru", 1, 3, tag=80), repeated, repeated]),
    )
    write(
        "ru_glossary.md",
        "# Глоссарий\n\n"
        + "\n\n".join(f"## Термин {i}\n\nКраткое определение номер {i}." for i in range(1, 12)),
    )
    legacy = "\n\n".join(paragraphs(rnd, "ru", 1, 4, tag=n) for n in range(90, 94))
    write("ru_legacy_cp1251.txt", legacy.encode("cp1251"))
    write("ru_report.pdf", pdf_report(rnd, "ru", "Отчёт по резервному копированию", 200))


def mixed(rnd: Lcg) -> None:
    intro = (
        "Клиент вызывает `GET /v1/items` и получает список; поле `next_cursor` указывает "
        "на следующую страницу."
    )
    sample = '{"items": [], "next_cursor": null}\n' * 4
    write(
        "mixed_api_notes.md",
        "\n\n".join(
            [
                "# Заметки по API",
                intro,
                f"## Пример\n\n```json\n{sample}```",
                md_section(rnd, "ru", 2, "Ограничения", 2, 100),
                "## Errors\n\nThe server answers `429` when the limit is exceeded; the client "
                "waits and repeats the call.",
            ]
        ),
    )
    write(
        "empty_headings.md",
        "# One\n\n## Two\n\n### Three\n\n#### Four\n\nOnly the last heading has text.\n\n## Five\n",
    )


def pdf_report(rnd: Lcg, lang: str, title: str, tag: int) -> bytes:
    """Five pages with a running title and page numbers around two wrapped paragraphs."""
    pages = []
    for number in range(1, 6):
        lines = [f"{title} {number}"]
        for part in range(2):
            text = " ".join(sentences(rnd, lang, 4, tag + number * 10 + part))
            lines.extend(textwrap.wrap(text, width=72))
        lines.append(f"{'Page' if lang == 'en' else 'Стр.'} {number}")
        pages.append(lines)
    return build_pdf(pages, title=title)


def main() -> None:
    english(Lcg(11))
    russian(Lcg(23))
    mixed(Lcg(31))


if __name__ == "__main__":
    main()
