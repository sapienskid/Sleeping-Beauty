"""In-process LanceDB store with native Tantivy BM25 Full-Text Search and Vector Hybrid Search."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import lancedb
from lancedb.index import FTS
from src.config import settings
from src.ingest.chunker import DocumentChunk


class KnowledgeStore:
    _instance: Optional["KnowledgeStore"] = None

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.LANCEDB_DIR
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))
        self.table_name = settings.TABLE_NAME
        self.table = None
        self._ensure_table()

    @classmethod
    def get_instance(cls) -> "KnowledgeStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_table(self):
        """Opens existing table or leaves table None until first insertion."""
        existing_tables = self.db.table_names()
        if self.table_name in existing_tables:
            self.table = self.db.open_table(self.table_name)
        else:
            self.table = None

    def add_chunks(self, chunks: List[DocumentChunk], vectors: List[List[float]]):
        """Add chunks and their vectors to the database, creating or updating the table & FTS index."""
        if not chunks:
            return

        records = []
        for chunk, vec in zip(chunks, vectors):
            records.append({
                "id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "file_path": chunk.file_path,
                "vault_name": chunk.vault_name,
                "relative_path": chunk.relative_path,
                "title": chunk.title,
                "section": chunk.section,
                "text": chunk.text,
                "vector": vec,
            })

        if self.table is None:
            self.table = self.db.create_table(self.table_name, data=records, mode="overwrite")
            # Create Full Text Search (BM25) index on the text column
            try:
                self.table.create_index("text", config=FTS())
            except Exception as e:
                print(f"Notice: Initial FTS index creation: {e}")
        else:
            # Cleanly replace existing records for this vault
            vaults_in_batch = list({chunk.vault_name for chunk in chunks if chunk.vault_name})
            for v in vaults_in_batch:
                try:
                    self.table.delete(f"vault_name = '{v}'")
                except Exception:
                    pass

            self.table.add(records)
            # Re-index FTS after addition
            try:
                self.table.create_index("text", config=FTS(), replace=True)
            except Exception:
                pass

    def hybrid_search(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 4,
        vault_name: Optional[str] = None,
        disabled_vaults: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid BM25 + Vector ANN search merged with Reciprocal Rank Fusion (RRF)."""
        if self.table is None:
            return []

        search_query = (
            self.table.search(
                query_type="hybrid",
                vector_column_name="vector",
                fts_columns="text",
            )
            .vector(query_vector)
            .text(query)
            .limit(top_k)
        )

        filter_clause = None
        if vault_name and vault_name != "all":
            filter_clause = f"vault_name = '{vault_name}'"
        elif disabled_vaults:
            escaped = ", ".join(f"'{v}'" for v in disabled_vaults if v)
            if escaped:
                filter_clause = f"vault_name NOT IN ({escaped})"

        if filter_clause:
            search_query = search_query.where(filter_clause)

        try:
            results = search_query.to_list()
        except Exception:
            # Fallback to vector search if FTS index is temporarily updating or unavailable
            vec_search = self.table.search(query_vector, vector_column_name="vector").limit(top_k)
            if filter_clause:
                vec_search = vec_search.where(filter_clause)
            results = vec_search.to_list()

        formatted = []
        for r in results:
            formatted.append({
                "id": r.get("id"),
                "doc_id": r.get("doc_id"),
                "file_path": r.get("file_path"),
                "vault_name": r.get("vault_name"),
                "relative_path": r.get("relative_path"),
                "title": r.get("title"),
                "section": r.get("section"),
                "text": r.get("text"),
                "score": r.get("_score", r.get("_distance", 0.0)),
            })

        return formatted

    def list_vaults(self) -> List[Dict[str, Any]]:
        """List all indexed vaults and statistics using PyArrow (zero pandas dependency)."""
        if self.table is None:
            return []

        try:
            arrow_tbl = self.table.to_arrow()
            if arrow_tbl.num_rows == 0:
                return []

            vault_col = arrow_tbl.column("vault_name").to_pylist()
            doc_col = arrow_tbl.column("doc_id").to_pylist()
            rel_col = arrow_tbl.column("relative_path").to_pylist()

            vault_map: Dict[str, Dict[str, Any]] = {}
            for v, doc, rel in zip(vault_col, doc_col, rel_col):
                if v not in vault_map:
                    vault_map[v] = {"chunks": 0, "docs": set(), "files": []}
                vault_map[v]["chunks"] += 1
                vault_map[v]["docs"].add(doc)
                if rel not in vault_map[v]["files"] and len(vault_map[v]["files"]) < 5:
                    vault_map[v]["files"].append(rel)

            stats = []
            for v_name, data in vault_map.items():
                stats.append({
                    "vault_name": v_name,
                    "total_chunks": data["chunks"],
                    "total_documents": len(data["docs"]),
                    "sample_files": data["files"],
                })
            return stats
        except Exception as e:
            print(f"Error listing vaults: {e}")
            return []

    def delete_vault(self, vault_name: str) -> bool:
        """Remove an entire vault from the store."""
        if self.table is None:
            return False

        try:
            self.table.delete(f"vault_name = '{vault_name}'")
            return True
        except Exception as e:
            print(f"Error deleting vault {vault_name}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get global database statistics."""
        if self.table is None:
            return {"total_chunks": 0, "total_documents": 0, "vaults": []}

        try:
            total_chunks = self.table.count_rows()
            vaults = self.list_vaults()
            total_docs = sum(v["total_documents"] for v in vaults)
            return {
                "total_chunks": total_chunks,
                "total_documents": total_docs,
                "vaults": vaults,
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"total_chunks": 0, "total_documents": 0, "vaults": []}
