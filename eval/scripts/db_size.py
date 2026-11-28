"""Where the database bytes go: size per table against the source text (NFR-02).

Ingests a generated corpus of natural-looking text through the real ``rag ingest`` with a hashing
embedder behind a mocked embedding server (vector size does not depend on content), then reports
the file size, the size of every table and index from ``dbstat``, and ratios to the source text.

    uv run python eval/scripts/db_size.py --megabytes 4 --dimension 384
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sqlite3
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import httpx

from rag_engine.cli.main import main as rag
from rag_engine.container import DefaultContainer
from rag_engine.testing import HashingEmbedder

DIGEST = "e" * 64


def corpus(root: Path, megabytes: float, seed: int) -> int:
    """Markdown files of about 20 KB with headings and Zipf-distributed words."""
    rng = random.Random(seed)  # noqa: S311 - reproducible test data

    def word(i: int) -> str:
        stem = rng.choice("bcdfghklmnprstvz") + rng.choice("aeiou") + rng.choice("bcdfgklmnprst")
        return f"{stem}{rng.choice('aeiou')}{rng.choice(['', 'n', 's', 'r', 'x'])}{i % 97}"

    vocab = [word(i) for i in range(6000)]
    weights = [1 / (rank + 1) for rank in range(len(vocab))]
    total = 0
    index = 0
    while total < megabytes * 1_000_000:
        parts = [f"# Document {index}\n"]
        for section in range(8):
            parts.append(f"\n## Section {section} {' '.join(rng.choices(vocab, weights, k=3))}\n")
            for _ in range(4):
                sentence = [" ".join(rng.choices(vocab, weights, k=rng.randint(8, 18))) + "."
                            for _ in range(rng.randint(3, 6))]  # fmt: skip
                parts.append(" ".join(sentence).capitalize() + "\n")
        text = "\n".join(parts)
        (root / f"doc{index:04d}.md").write_text(text, encoding="utf-8")
        total += len(text.encode())
        index += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megabytes", type=float, default=4.0)
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument(
        "--from-dir", type=Path, default=None, help="copy real Markdown files from here instead"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    embedder = HashingEmbedder(args.dimension)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "test", "digest": DIGEST}]})
        texts = json.loads(request.content)["input"]
        with ThreadPoolExecutor(max_workers=1) as pool:
            vectors = pool.submit(asyncio.run, embedder.embed_documents(texts)).result()
        return httpx.Response(200, json={"embeddings": vectors.tolist()})

    real = httpx.AsyncClient
    httpx.AsyncClient = lambda **kw: real(  # type: ignore[misc,assignment]
        base_url=kw["base_url"], transport=httpx.MockTransport(handler)
    )
    with tempfile.TemporaryDirectory(prefix="rag-size-") as folder:
        work = Path(folder)
        root = work / "notes"
        root.mkdir()
        if args.from_dir is not None:
            source_bytes = 0
            for n, path in enumerate(sorted(args.from_dir.rglob("*.md"))):
                if path.stat().st_size < 2048:
                    continue
                if source_bytes >= args.megabytes * 1_000_000:
                    break
                target = root / f"real{n:05d}.md"
                target.write_bytes(path.read_bytes())
                source_bytes += target.stat().st_size
        else:
            source_bytes = corpus(root, args.megabytes, seed=7)
        config = work / "rag.toml"
        config.write_text(
            '[parsing]\nisolate = []\n[embedding]\nenabled = true\nprovider = "ollama"\n'
            f'model = "test"\nrevision = "{DIGEST}"\ndimension = {args.dimension}\n'
            'base_url = "http://test"\n'
        )
        db = work / "rag.db"
        container = DefaultContainer()
        prefix = ["--config", str(config), "--db", str(db)]
        for step in (["source", "add", "notes", str(root)], ["ingest", "notes", "--json"]):
            if rag([*prefix, *step], container=container) != 0:
                with closing(sqlite3.connect(db)) as conn:
                    for key, code in conn.execute(
                        "SELECT d.source_key, j.error_code FROM jobs AS j"
                        " JOIN documents AS d ON d.id = j.document_id WHERE j.status = 'dead'"
                    ):
                        print("dead job:", key, code)
                raise SystemExit(f"rag {step[0]} failed")
        with closing(sqlite3.connect(db)) as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("VACUUM")
            pages = dict(
                conn.execute("SELECT name, SUM(pgsize) FROM dbstat GROUP BY name ORDER BY 2 DESC")
            )
            columns = {
                name: conn.execute(
                    f"SELECT COALESCE(SUM(length({expr})), 0) FROM chunks{where}"  # noqa: S608
                ).fetchone()[0]
                for name, expr, where in (
                    ("leaf_text", "text", " WHERE level = 0"),
                    ("parent_text", "text", " WHERE level = 1"),
                    ("body_an", "body_an", ""),
                    ("headings_an", "headings_an", ""),
                    ("heading_path_json", "heading_path_json", ""),
                    ("ids_and_hashes", "id || text_hash || COALESCE(embedding_input_hash, '')", ""),
                )
            }
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            vectors = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        size = db.stat().st_size
    report = {
        "source_bytes": source_bytes,
        "db_bytes": size,
        "ratio_total": round(size / source_bytes, 2),
        "chunks": chunks,
        "vectors": vectors,
        "vector_bytes_theory": vectors * args.dimension * 2,
        "ratio_without_vectors": round((size - vectors * args.dimension * 2) / source_bytes, 2),
        "chunk_columns_x_source": {k: round(v / source_bytes, 2) for k, v in columns.items()},
        "by_object": {
            k: {"bytes": v, "x_source": round(v / source_bytes, 2)}
            for k, v in list(pages.items())[:12]
        },
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
