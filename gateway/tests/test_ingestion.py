"""Tests for ingestion pipeline and /ingest, /collections endpoints."""

from __future__ import annotations

import pytest
import httpx


pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.fixture
def sample_txt(tmp_path) -> str:
    """Create a sample text file and return its path."""
    doc = tmp_path / "sample.txt"
    doc.write_text(
        "Arch Linux uses pacman as its package manager.\n\n"
        "The key command for updating is pacman -Syu which synchronizes\n"
        "repositories and upgrades all packages. You should always do a\n"
        "full system upgrade, never a partial one.\n\n"
        "AUR helpers like yay and paru make installing community packages easy."
    )
    return str(doc)


@pytest.fixture
def sample_md(tmp_path) -> str:
    """Create a sample markdown file."""
    doc = tmp_path / "guide.md"
    doc.write_text(
        "# Docker Compose Guide\n\n"
        "## Services\n\n"
        "Docker Compose uses a YAML file to define multi-container applications.\n"
        "The `docker compose up` command starts all services defined in the file.\n\n"
        "## Volumes\n\n"
        "Named volumes persist data across container restarts.\n"
        "Bind mounts map host directories into containers.\n"
    )
    return str(doc)


async def test_ingest_file(client: httpx.AsyncClient, sample_txt: str):
    """Ingest a single text file → chunks appear in collection."""
    resp = await client.post(
        "/ingest",
        json={
            "source_path": sample_txt,
            "collection": "project-docs",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["chunks_created"] >= 1
    assert data["collection"] == "project-docs"
    assert "job_id" in data


async def test_ingest_collections_count(client: httpx.AsyncClient, sample_txt: str):
    """After ingestion, /collections shows the correct count."""
    # Ingest first
    await client.post(
        "/ingest",
        json={"source_path": sample_txt, "collection": "test-collection"},
    )

    resp = await client.get("/collections")
    assert resp.status_code == 200
    data = resp.json()
    found = [c for c in data["collections"] if c["name"] == "test-collection"]
    assert len(found) == 1
    assert found[0]["count"] >= 1


async def test_ingest_dedup_skips(client: httpx.AsyncClient, sample_txt: str):
    """Ingesting the same file twice without force_reindex → skipped."""
    # First ingest
    resp1 = await client.post(
        "/ingest",
        json={"source_path": sample_txt, "collection": "dedup-test"},
    )
    assert resp1.json()["status"] == "completed"
    first_count = resp1.json()["chunks_created"]

    # Second ingest (same file, unchanged)
    resp2 = await client.post(
        "/ingest",
        json={"source_path": sample_txt, "collection": "dedup-test"},
    )
    assert resp2.json()["status"] == "skipped"
    assert resp2.json()["chunks_created"] == 0


async def test_ingest_force_reindex(client: httpx.AsyncClient, sample_txt: str):
    """force_reindex=true re-ingests even if unchanged."""
    # First ingest
    await client.post(
        "/ingest",
        json={"source_path": sample_txt, "collection": "force-test"},
    )
    # Force re-ingest
    resp = await client.post(
        "/ingest",
        json={
            "source_path": sample_txt,
            "collection": "force-test",
            "force_reindex": True,
        },
    )
    assert resp.json()["status"] == "completed"
    assert resp.json()["chunks_created"] >= 1


async def test_ingest_directory(
    client: httpx.AsyncClient, sample_txt: str, sample_md: str, tmp_path
):
    """Ingest a directory with multiple files."""
    resp = await client.post(
        "/ingest",
        json={
            "source_path": str(tmp_path),
            "collection": "dir-test",
            "recursive": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("completed", "completed_with_errors")
    assert data["chunks_created"] >= 2
    assert data["files_processed"] >= 2


async def test_ingest_nonexistent_file(client: httpx.AsyncClient):
    """Ingesting a nonexistent file → 400."""
    resp = await client.post(
        "/ingest",
        json={"source_path": "/nonexistent/file.txt", "collection": "test"},
    )
    assert resp.status_code == 400
    assert "not found" in resp.json()["error"].lower()


async def test_ingest_unsupported_format(client: httpx.AsyncClient, tmp_path):
    """Ingesting an unsupported format → 500 with error."""
    f = tmp_path / "data.csv"
    f.write_text("a,b,c\n1,2,3")
    resp = await client.post(
        "/ingest",
        json={"source_path": str(f), "collection": "test"},
    )
    assert resp.status_code == 500
    assert "unsupported" in resp.json()["error"].lower()
