"""The `corpus_mini` documents (Russian, English; Markdown, HTML, text) through the pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_engine.chunking.registry import ChunkerRegistry
from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.domain import ChunkPolicy, SourceKey
from rag_engine.ingestion.processing import DocumentProcessor, ProcessedDocument
from rag_engine.parsing.registry import default_parsers
from rag_engine.ports import ParseLimits
from rag_engine.testing import FakeClock
from tests.chunking_checks import SMALL_POLICY, check_chunks
from tests.factories import make_ids

CORPUS = Path(__file__).parents[1] / "fixtures" / "corpus_mini"
COUNTER = HeuristicTokenCounter()
VERSION = make_ids().document_version_id()
LIMITS = ParseLimits()
DEFAULT = ChunkPolicy()

# file name -> (parser, chunking strategy chosen automatically)
EXPECTED = {
    "empty_headings.md": ("md", "structure_aware"),
    "en_backup_guide.md": ("md", "structure_aware"),
    "en_changelog.md": ("md", "structure_aware"),
    "en_config_reference.md": ("md", "structure_aware"),
    "en_faq.txt": ("txt", "recursive"),
    "en_handbook.docx": ("docx", "structure_aware"),
    "en_release_process.md": ("md", "structure_aware"),
    "en_report.pdf": ("pdf", "recursive"),
    "en_single_paragraph.txt": ("txt", "fixed"),
    "en_status_page.html": ("html", "structure_aware"),
    "mixed_api_notes.md": ("md", "structure_aware"),
    "ru_faq.md": ("md", "structure_aware"),
    "ru_glossary.md": ("md", "structure_aware"),
    "ru_handbook.docx": ("docx", "structure_aware"),
    "ru_instruction.md": ("md", "structure_aware"),
    "ru_legacy_cp1251.txt": ("txt", "recursive"),
    "ru_page.html": ("html", "structure_aware"),
    "ru_report.pdf": ("pdf", "recursive"),
    "ru_policy.txt": ("txt", "recursive"),
    "ru_repeated_notes.txt": ("txt", "recursive"),
}


def process(name: str, policy: ChunkPolicy = DEFAULT) -> ProcessedDocument:
    runner = DocumentProcessor(default_parsers(), FakeClock(), ChunkerRegistry(COUNTER))
    return runner.process((CORPUS / name).read_bytes(), SourceKey(name), LIMITS, policy, VERSION)


def test_the_manifest_lists_every_file_of_the_corpus() -> None:
    assert sorted(p.name for p in CORPUS.iterdir()) == sorted(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
@pytest.mark.parametrize("policy", [DEFAULT, SMALL_POLICY], ids=["default", "small"])
def test_every_document_is_parsed_and_chunked_within_the_invariants(
    name: str, policy: ChunkPolicy
) -> None:
    done = process(name, policy)
    parser, strategy = EXPECTED[name]
    assert (done.parsed.parser_id, done.chunker_id) == (parser, strategy)
    check_chunks(done.parsed.document, done.chunks, policy, COUNTER, done.chunker_fingerprint)
    leaves = [c for c in done.chunks if c.level == 0]
    assert leaves and all(c.token_count <= policy.max_tokens for c in leaves)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_processing_a_document_twice_gives_identical_chunks(name: str) -> None:
    assert process(name).chunks == process(name).chunks


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_the_language_of_chunks_follows_the_text(name: str) -> None:
    done = process(name)
    languages = [c.language.value for c in done.chunks if c.level == 0]
    for prefix, language in (("ru_", "ru"), ("en_", "en")):
        if name.startswith(prefix):
            assert done.parsed.document.language.value == language
            assert languages.count(language) >= len(languages) / 2


def test_a_legacy_encoding_is_decoded_to_readable_text() -> None:
    done = process("ru_legacy_cp1251.txt")
    assert done.parsed.mime.encoding == "cp1251"
    assert any("резервное копирование" in c.text or "Планировщик" in c.text for c in done.chunks)
    assert not any("�" in c.text for c in done.chunks)


def test_html_pages_lose_their_furniture_and_decode_entities_once() -> None:
    done = process("en_status_page.html")
    text = "\n".join(c.text for c in done.chunks)
    assert "Copyright notice" not in text and "Home" not in text
    assert "running & healthy" in text and "&amp;" not in text
    assert done.chunks[0].heading_path[0] == "Service status"


def test_nested_headings_give_chunks_their_heading_path() -> None:
    paths = {c.heading_path for c in process("en_backup_guide.md").chunks if c.level == 0}
    assert ("Backup guide", "Preparing the volume", "Checking free space") in paths
    assert ("Backup guide", "Settings") in paths


def test_tiny_sections_are_merged_into_one_chunk() -> None:
    (changelog,) = [c for c in process("en_changelog.md").chunks if c.level == 0]
    assert "Version 1.0" in changelog.text and "Version 1.8" in changelog.text
    (headings,) = [c for c in process("empty_headings.md").chunks if c.level == 0]
    assert headings.text.startswith("One") and "Only the last heading" in headings.text


def test_sections_with_several_leaves_have_a_parent_that_contains_them() -> None:
    chunks = list(process("en_backup_guide.md").chunks)
    parents = {c.id: c for c in chunks if c.level == 1}
    assert parents
    for leaf in chunks:
        if leaf.level == 0 and leaf.parent_id is not None:
            parent = parents[leaf.parent_id]
            assert parent.char_start <= leaf.char_start and leaf.char_end <= parent.char_end
            assert not parent.indexed


def test_a_code_block_is_kept_whole_when_it_fits() -> None:
    chunks = process("en_backup_guide.md").chunks
    assert any("def run_backup" in c.text and "return verify(target)" in c.text for c in chunks)


@pytest.mark.parametrize("name", ["en_report.pdf", "ru_report.pdf"])
def test_pdf_reports_keep_their_pages_and_lose_running_titles(name: str) -> None:
    done = process(name)
    text = done.parsed.document.normalized_text
    assert done.parsed.document.title and "headers_footers_removed" in done.parsed.document.warnings
    for furniture in ("Backup report", "Отчёт по резервному копированию", "Page ", "Стр."):
        assert furniture not in text
    pages = [c.page_start for c in done.chunks if c.level == 0]
    assert all(page is not None for page in pages) and pages == sorted(p for p in pages if p)
    assert min(p for p in pages if p) == 1 and max(p for p in pages if p) == 5
    assert all(len(b.text) > 100 for b in done.parsed.document.blocks)  # joined, not line by line


@pytest.mark.parametrize("name", ["en_handbook.docx", "ru_handbook.docx"])
def test_docx_handbooks_keep_structure_tables_and_metadata(name: str) -> None:
    done = process(name)
    document = done.parsed.document
    kinds = {b.kind.value for b in document.blocks}
    assert {"heading", "paragraph", "list_item", "table", "quote", "code"} <= kinds
    assert document.author and document.title
    paths = {c.heading_path for c in done.chunks if c.level == 0}
    assert any(len(path) == 3 for path in paths)  # title > heading > subheading
    table = next(b for b in document.blocks if b.kind.value == "table")
    assert table.text.splitlines()[1] == "option_1 | 10"
    assert any(table.text in c.text for c in done.chunks if c.level == 0)
