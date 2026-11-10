"""Command line interface for Sleeping Beauty."""

from pathlib import Path
from typing import Optional
import asyncio
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from src.config import settings

app = typer.Typer(
    name="sb",
    help="Sleeping Beauty: High-performance local knowledge search & RAG engine.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def index(
    vault_path: str = typer.Argument(..., help="Path to the directory or vault to index"),
    chunk_size: int = typer.Option(settings.CHUNK_SIZE, "--chunk-size", "-c", help="Chunk size in characters"),
    chunk_overlap: int = typer.Option(settings.CHUNK_OVERLAP, "--overlap", "-o", help="Chunk overlap in characters"),
):
    """Scan and index all files in a vault or directory."""
    from src.web.app import run_indexing

    p = Path(vault_path).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        console.print(f"[bold red]Error:[/bold red] Directory '{vault_path}' does not exist.")
        raise typer.Exit(code=1)

    with console.status(f"[bold green]Scanning and indexing vault: {p.name}...[/bold green]"):
        result = run_indexing(p, chunk_size, chunk_overlap)

    console.print(
        Panel.fit(
            f"[bold green]Indexing Complete![/bold green]\n"
            f"Vault: [cyan]{result['vault_name']}[/cyan]\n"
            f"Files Scanned: [yellow]{result['files_scanned']}[/yellow]\n"
            f"Chunks Embedded & Indexed: [yellow]{result['chunks_indexed']}[/yellow]\n"
            f"Storage: [dim]{settings.LANCEDB_DIR}[/dim]",
            title="Success",
            border_style="green",
        )
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(settings.RAG_TOP_K, "--top-k", "-k", help="Number of results to retrieve"),
    vault: Optional[str] = typer.Option(None, "--vault", "-v", help="Filter by specific vault name"),
):
    """Run hybrid semantic & keyword search across indexed knowledge."""
    from src.rag.engine import RAGEngine

    engine = RAGEngine()
    with console.status("[bold blue]Performing hybrid search (BM25 + Dense Vectors)...[/bold blue]"):
        results = engine.retrieve(query, top_k=top_k, vault_name=vault)

    if not results:
        console.print("[yellow]No matching documents found.[/yellow]")
        return

    table = Table(title=f"Hybrid Search Results for '{query}'", border_style="cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Score", style="green", width=8)
    table.add_column("Document / Section", style="bold cyan")
    table.add_column("Snippet", style="white")

    for i, r in enumerate(results, 1):
        rel = r.get("relative_path", "")
        sec = f" > {r.get('section')}" if r.get("section") else ""
        text = r.get("text", "").replace("\n", " ")
        if len(text) > 120:
            text = text[:120] + "..."
        score = f"{r.get('score', 0.0):.2f}"
        table.add_row(str(i), score, f"{rel}{sec}", text)

    console.print(table)


@app.command()
def ask(
    query: str = typer.Argument(..., help="Question to ask the local RAG engine"),
    top_k: int = typer.Option(settings.RAG_TOP_K, "--top-k", "-k", help="Number of chunks to retrieve"),
    vault: Optional[str] = typer.Option(None, "--vault", "-v", help="Filter by vault name"),
):
    """Ask a question; retrieved context is passed to the local 1-2B micro LLM."""
    from src.rag.engine import RAGEngine

    engine = RAGEngine()

    async def _run():
        chunks = engine.retrieve(query, top_k=top_k, vault_name=vault)
        if chunks:
            console.print(f"[dim]Retrieved {len(chunks)} relevant snippets.[/dim]")
            for idx, c in enumerate(chunks, 1):
                console.print(f"  [dim][{idx}][/dim] [cyan]{c.get('relative_path')}[/cyan] (Score: {c.get('score', 0.0):.2f})")
            console.print()

        console.print(f"[bold cyan]Answer:[/bold cyan]")
        async for chunk in engine.ask_stream(query, top_k=top_k, vault_name=vault):
            if chunk.get("type") == "token":
                print(chunk.get("token"), end="", flush=True)
            elif chunk.get("type") == "error":
                console.print(f"\n[bold red]{chunk.get('error')}[/bold red]")
        print("\n")

    asyncio.run(_run())


@app.command()
def chat(
    prompt: Optional[str] = typer.Argument(None, help="Direct prompt to the LLM (without retrieval)"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Sampling temperature"),
    max_tokens: int = typer.Option(512, "--max-tokens", "-m", help="Maximum tokens to generate"),
):
    """Direct chat with the local micro LLM (without vector retrieval)."""
    import httpx

    endpoint = f"{settings.LLAMA_SERVER_URL.rstrip('/')}/chat/completions"

    async def _send(user_msg: str):
        payload = {
            "model": settings.LLM_MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a helpful, brilliant AI assistant."},
                {"role": "user", "content": user_msg},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {settings.LLAMA_API_KEY}"}
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", endpoint, json=payload, headers=headers) as response:
                    if response.status_code != 200:
                        console.print(f"[bold red]LLM Error {response.status_code}[/bold red]")
                        return
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            import json
                            data = json.loads(data_str)
                            token = data.get("choices", [{}])[0].get("delta", {}).get("content") or ""
                            if token:
                                print(token, end="", flush=True)
                        except Exception:
                            continue
            print("\n")
        except httpx.ConnectError:
            console.print(f"[bold red]Error: llama-server is not running at {settings.LLAMA_SERVER_URL}[/bold red]")

    if prompt:
        console.print(f"[bold cyan]User:[/bold cyan] {prompt}")
        console.print(f"[bold green]LLM (Qwen 2.5 1.5B):[/bold green]")
        asyncio.run(_send(prompt))
    else:
        console.print(Panel.fit("[bold green]Direct LLM Chat (Type 'exit' to quit)[/bold green]", border_style="cyan"))
        while True:
            try:
                user_input = console.input("[bold cyan]> [/bold cyan]")
                if user_input.strip().lower() in ["exit", "quit", "q"]:
                    break
                if not user_input.strip():
                    continue
                console.print(f"[bold green]LLM:[/bold green]")
                asyncio.run(_send(user_input))
            except (KeyboardInterrupt, EOFError):
                break


@app.command("start-llm")
def start_llm():
    """Start local llama-server for generation LLM (Qwen 2.5 1.5B on port 10000)."""
    import subprocess
    import httpx

    try:
        r = httpx.get(f"{settings.LLAMA_SERVER_URL.rstrip('/')}/models", timeout=1.0)
        if r.status_code == 200:
            console.print("[green]llama-server (LLM) is already running on port 10000.[/green]")
            return
    except Exception:
        pass

    bin_path = settings.LLAMA_SERVER_BIN
    model_path = Path(settings.MODELS_DIR) / "qwen2.5-1.5b-instruct-q8_0.gguf"
    if not model_path.exists():
        console.print(f"[red]Error: Model not found at {model_path}[/red]")
        return

    cmd = [
        str(bin_path),
        "--host", "0.0.0.0",
        "--port", "10000",
        "-m", str(model_path),
        "-ngl", "99",
        "-fa", "on",
        "-c", "4096",
        "--jinja",
    ]
    console.print("[cyan]Launching LLM llama-server on port 10000...[/cyan]")
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    console.print("[bold green]LLM llama-server launched in background (http://127.0.0.1:10000)![/bold green]")


@app.command("start-embed")
def start_embed():
    """Start local llama-server for EmbeddingGemma-2 embeddings (port 10001)."""
    import subprocess
    import httpx

    server_url = settings.EMBEDDING_SERVER_URL or "http://127.0.0.1:10001/v1"
    try:
        r = httpx.get(f"{server_url.rstrip('/')}/models", timeout=1.0)
        if r.status_code == 200:
            console.print("[green]llama-server (EmbeddingGemma-2) is already running on port 10001.[/green]")
            return
    except Exception:
        pass

    bin_path = settings.LLAMA_SERVER_BIN
    model_path = settings.EMBEDDING_GGUF_PATH
    if not model_path.exists():
        console.print(f"[red]Error: GGUF model not found at {model_path}[/red]")
        return

    cmd = [
        str(bin_path),
        "--host", "0.0.0.0",
        "--port", "10001",
        "-m", str(model_path),
        "--embedding",
        "-ngl", "99",
        "-c", "2048",
    ]
    console.print("[cyan]Launching EmbeddingGemma-2 llama-server on port 10001...[/cyan]")
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    console.print("[bold green]EmbeddingGemma-2 server launched in background (http://127.0.0.1:10001)![/bold green]")


@app.command("start-all")
def start_all():
    """Start both LLM and EmbeddingGemma servers in background."""
    start_llm()
    start_embed()


@app.command("stop-all")
def stop_all():
    """Stop all running llama-server processes."""
    import subprocess
    cmd = "pkill -f llama-server || true"
    subprocess.run(cmd, shell=True)
    console.print("[yellow]Stopped all llama-server background instances.[/yellow]")


@app.command()
def stats():
    """Show knowledge base statistics."""
    from src.db.store import KnowledgeStore

    store = KnowledgeStore.get_instance()
    st = store.get_stats()

    console.print(
        Panel.fit(
            f"Total Chunks: [yellow]{st['total_chunks']}[/yellow]\n"
            f"Total Documents: [yellow]{st['total_documents']}[/yellow]\n"
            f"Indexed Vaults: [cyan]{len(st['vaults'])}[/cyan]\n"
            f"Storage Path: [dim]{settings.LANCEDB_DIR}[/dim]",
            title="Sleeping Beauty Stats",
            border_style="blue",
        )
    )

    if st["vaults"]:
        table = Table(title="Vaults", border_style="blue")
        table.add_column("Vault Name", style="bold cyan")
        table.add_column("Docs", style="green")
        table.add_column("Chunks", style="yellow")
        table.add_column("Sample Files", style="dim")

        for v in st["vaults"]:
            sample = ", ".join(v.get("sample_files", [])[:3])
            table.add_row(v["vault_name"], str(v["total_documents"]), str(v["total_chunks"]), sample)

        console.print(table)


@app.command()
def serve(
    host: str = typer.Option(settings.HOST, "--host", "-h", help="Host to bind"),
    port: int = typer.Option(settings.PORT, "--port", "-p", help="Port to bind"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
):
    """Start the Sleeping Beauty Web Dashboard & API server."""
    import uvicorn

    console.print(
        Panel.fit(
            f"[bold green]Starting Sleeping Beauty Hub[/bold green]\n"
            f"Web Dashboard: [bold cyan]http://{host}:{port}[/bold cyan]\n"
            f"API Docs:      [dim]http://{host}:{port}/docs[/dim]\n"
            f"LLM Endpoint:  [dim]{settings.LLAMA_SERVER_URL}[/dim]",
            title="Server Online",
            border_style="green",
        )
    )
    uvicorn.run("src.web.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
