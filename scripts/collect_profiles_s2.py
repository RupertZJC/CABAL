#!/usr/bin/env python3
"""
Collect researcher profiles from Semantic Scholar API.

Uses disambiguated author IDs — each profile = exactly ONE real researcher.
Rate limit: 1 req/sec (free tier). Optimized with batch endpoints.

Usage:
    export S2_API_KEY="s2k-..."
    python scripts/collect_profiles_s2.py --areas cs.CV --num-authors 10
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────

TARGET_AREAS = {
    "cs.CV": {
        "search_queries": [
            "object detection", "image segmentation",
            "image generation diffusion", "3D reconstruction NeRF",
            "video understanding action recognition",
            "visual question answering",
            "pose estimation human",
            "domain adaptation generalization",
            "self-supervised representation learning",
            "adversarial robustness attack",
            "image restoration super-resolution",
            "multi-modal vision language",
        ],
        "s2_fields": ["Computer Vision", "Computer Science"],
    },
    "cs.CL": {
        "search_queries": [
            "large language model training",
            "machine translation multilingual",
            "question answering reasoning",
            "text summarization generation",
            "information extraction named entity",
            "dialogue conversational system",
            "retrieval augmented generation",
            "text classification sentiment",
            "code generation programming",
            "instruction tuning alignment",
            "few-shot in-context learning",
            "language model evaluation benchmark",
        ],
        "s2_fields": [
            "Natural Language Processing", "Computer Science",
        ],
    },
    "cs.LG": {
        "search_queries": [
            "deep learning optimization",
            "generalization theory neural networks",
            "federated learning privacy",
            "reinforcement learning policy",
            "graph neural networks",
            "meta learning few shot",
            "continual learning forgetting",
            "uncertainty estimation calibration",
            "implicit neural representations",
            "distribution shift robustness",
            "neural architecture search",
            "learning theory optimization",
        ],
        "s2_fields": [
            "Machine Learning", "Artificial Intelligence", "Computer Science",
        ],
    },
    "cs.AI": {
        "search_queries": [
            "knowledge graph reasoning",
            "planning automated decision making",
            "multi agent systems cooperation",
            "causal inference discovery",
            "symbolic reasoning neural",
            "search optimization algorithms",
            "game playing agents",
            "constraint satisfaction optimization",
            "commonsense reasoning",
            "explainable artificial intelligence",
            "bayesian inference probabilistic",
            "recommender systems",
        ],
        "s2_fields": [
            "Artificial Intelligence", "Computer Science",
        ],
    },
}

STOP_WORDS = {
    "this", "that", "with", "from", "have", "been", "which", "their",
    "these", "those", "about", "based", "using", "paper", "method",
    "approach", "model", "propose", "proposed", "problem", "result",
    "show", "demonstrate", "experiment", "performance", "present",
    "introduce", "achieve", "improve", "compare", "state", "existing",
    "various", "well", "also", "first", "novel", "without", "across",
    "however", "through", "between", "include", "including",
    "different", "several", "many", "large", "small", "recent",
    "previous", "better", "higher", "lower", "both",
    "learning", "training", "data", "dataset", "datasets",
    "image", "images", "video", "videos", "text", "task", "tasks",
    "used", "using", "results", "result", "the", "and", "for",
    "are", "but", "not", "you", "all", "can", "had", "her", "was",
    "our", "some", "than", "its", "has", "been", "will", "each",
    "were", "said", "does", "how", "when", "what",
}


# ── Anonymization ────────────────────────────────────────────────────────

class Anonymizer:
    def __init__(self):
        self._counter = 0
        self._map: dict[str, str] = {}

    def anon(self, key: str) -> str:
        if key not in self._map:
            self._counter += 1
            self._map[key] = f"Researcher_{self._counter:03d}"
        return self._map[key]

    @staticmethod
    def profile_id(author_id: str) -> str:
        h = hashlib.sha256(f"malbid:{author_id}".encode()).hexdigest()[:12]
        return f"U_{h}"

    @staticmethod
    def clean(text: str) -> str:
        return re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[EMAIL]', text)


# ── S2 API Client ────────────────────────────────────────────────────────

class S2Client:
    """Thin wrapper around Semantic Scholar API with rate limiting."""

    BASE = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._last_req = 0.0
        self._req_count = 0

    def _wait(self):
        """Enforce 1 req/sec rate limit."""
        elapsed = time.time() - self._last_req
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_req = time.time()
        self._req_count += 1

    def _get(self, endpoint: str, params: dict = None) -> dict:
        """GET request with rate limiting + retry."""
        import requests
        self._wait()

        url = f"{self.BASE}/{endpoint}"
        headers = {"x-api-key": self.api_key}
        params = params or {}

        for attempt in range(4):
            try:
                resp = requests.get(
                    url, params=params, headers=headers, timeout=30,
                )
                if resp.status_code == 429:
                    logger.debug("    Rate limited, waiting 5s...")
                    time.sleep(5)
                    continue
                if resp.status_code in (500, 502, 503, 504):
                    logger.debug("    Server error %d, retrying...", resp.status_code)
                    time.sleep(5)
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if attempt < 3:
                    time.sleep(2)
                else:
                    logger.warning("    Request failed after retries: %s", e)
                    return {}
        return {}

    def search_papers(
        self, query: str, limit: int = 100, fields_of_study: list[str] = None,
    ) -> list[dict]:
        """Search papers by keyword, return list with full metadata."""
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,abstract,year,authors,externalIds,"
                     "publicationVenue,citationCount,fieldsOfStudy",
        }
        result = self._get("paper/search", params)
        papers = result.get("data", [])

        # Flatten: add author_name for convenience
        out = []
        for p in papers:
            authors = []
            for a in p.get("authors", []):
                authors.append({
                    "author_id": str(a.get("authorId", "")),
                    "name": a.get("name", ""),
                })
            out.append({
                "title": p.get("title", ""),
                "abstract": p.get("abstract", "") or "",
                "authors": authors,
                "year": p.get("year", 2024) or 2024,
                "s2_paper_id": p.get("paperId", ""),
                "venue": (
                    p.get("publicationVenue", {}) or {}
                ).get("name", "") or p.get("venue", "") or "",
                "fields_of_study": [
                    f.get("name", f) if isinstance(f, dict) else str(f)
                    for f in (p.get("fieldsOfStudy", []) or [])
                ],
                "citation_count": p.get("citationCount", 0) or 0,
            })

        # Filter to English + complete info only
        out = [p for p in out if has_complete_info(p)]
        return out

    def get_author_papers(
        self, author_id: str, limit: int = 30,
    ) -> list[dict]:
        """Get all papers for a disambiguated author."""
        params = {
            "limit": limit,
            "fields": "title,abstract,year,authors,externalIds,"
                     "publicationVenue,citationCount,fieldsOfStudy",
        }
        result = self._get(f"author/{author_id}/papers", params)
        papers = result.get("data", [])

        out = []
        for p in papers:
            authors = []
            for a in p.get("authors", []):
                authors.append({
                    "author_id": str(a.get("authorId", "")),
                    "name": a.get("name", ""),
                })
            out.append({
                "title": p.get("title", ""),
                "abstract": p.get("abstract", "") or "",
                "authors": authors,
                "year": p.get("year", 2024) or 2024,
                "s2_paper_id": p.get("paperId", ""),
                "venue": (
                    p.get("publicationVenue", {}) or {}
                ).get("name", "") or "",
                "fields_of_study": [
                    f.get("name", f) if isinstance(f, dict) else str(f)
                    for f in (p.get("fieldsOfStudy", []) or [])
                ],
                "citation_count": p.get("citationCount", 0) or 0,
            })

        # Filter to English + complete info only
        out = [p for p in out if has_complete_info(p)]
        return out

    def get_batch_paper_details(
        self, paper_ids: list[str],
    ) -> dict[str, dict]:
        """Batch fetch paper details (abstracts). Max 500 IDs per call."""
        if not paper_ids:
            return {}

        # Process in batches of 500
        all_details = {}
        for i in range(0, len(paper_ids), 500):
            batch = paper_ids[i:i + 500]
            params = {
                "fields": "title,abstract,year,publicationVenue,citationCount",
            }
            result = self._get("paper/batch", {
                **params,
                "ids": ",".join(batch),
            })
            for p in (result.get("data", []) or []):
                if p:
                    pid = p.get("paperId", "")
                    if pid:
                        all_details[pid] = p
        return all_details


# ── Language Filter ──────────────────────────────────────────────────────

def is_english(text: str) -> bool:
    """
    Quick check: >90% ASCII characters = English.
    Catches non-English papers (Chinese, Japanese, etc.) without heavy deps.
    """
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / max(1, len(text))) > 0.90


def has_complete_info(paper: dict) -> bool:
    """
    A paper is "complete" if it has:
      - Non-empty title
      - English title + abstract
      - Abstract with >50 chars (real content, not a stub)
      - Valid year
    """
    title = paper.get("title", "") or ""
    abstract = paper.get("abstract", "") or ""
    if not title:
        return False
    if not abstract or len(abstract.strip()) < 50:
        return False
    if not is_english(title) or not is_english(abstract):
        return False
    return True


# ── Profile Building ─────────────────────────────────────────────────────

def build_profiles(
    author_papers: dict[str, list[dict]],
    area_name: str,
    max_pubs: int = 10,
    anonymizer: Anonymizer = None,
) -> list[dict]:
    """Build anonymized profiles from authorId→papers mapping."""
    anon = anonymizer or Anonymizer()
    profiles = []
    assigned_ids: set = set()

    for author_id, papers in sorted(
        author_papers.items(),
        key=lambda x: len(x[1]), reverse=True,
    ):
        if len(papers) < 2:
            continue

        # Deduplicate & sort — only keep complete papers
        seen_titles = set()
        unique = []
        for p in sorted(papers, key=lambda x: x["year"], reverse=True):
            if not has_complete_info(p):
                continue
            pid = p.get("s2_paper_id", "")
            t = p["title"].lower().strip()
            if pid and pid in assigned_ids:
                continue
            if t in seen_titles:
                continue
            seen_titles.add(t)
            if pid:
                assigned_ids.add(pid)
            unique.append(p)

        unique = unique[:max_pubs]
        if len(unique) < 2:
            continue

        # Keywords
        all_text = " ".join(
            (p["title"] + " " + (p.get("abstract", "") or "")[:500])
            for p in unique
        ).lower()
        words = re.findall(r'\b[a-z]{4,}\b', all_text)
        freq = Counter(w for w in words if w not in STOP_WORDS)
        keywords = [w for w, _ in freq.most_common(20)]

        # Publications
        publications = []
        for p in unique:
            abstract = p.get("abstract", "") or ""
            publications.append({
                "title": Anonymizer.clean(p["title"]),
                "abstract": Anonymizer.clean(abstract),
                "venue": p.get("venue", "Unknown"),
                "year": p["year"],
                "keywords": keywords[:6],
                "venue_type": "conference",
                "citation_count": p.get("citation_count", 0),
            })

        profile_id = anon.profile_id(author_id)
        profiles.append({
            "profile_id": profile_id,
            "research_areas": [area_name] + keywords[:6],
            "sub_areas": keywords[:5],
            "publications": publications,
            "h_index_estimate": min(int(len(unique) ** 0.5 * 4), 80),
            "total_papers_estimate": len(unique),
            "institutions": [],
            "coauthor_ids": [],
            "source_type": "semantic_scholar",
            "source_timestamp": datetime.now().isoformat(),
        })

    return profiles


# ── Main ─────────────────────────────────────────────────────────────────

def collect_area_round(
    client: S2Client,
    area_name: str,
    cfg: dict,
    min_papers: int,
    papers_per_query: int,
    max_pubs: int,
    anonymizer: Anonymizer,
    seen_author_ids: set,
    need_count: int,
) -> tuple[list[dict], int, int]:
    """
    Run one collection round for a single area.

    A "round" = search → aggregate authors → rank → fetch papers → build
    profiles. It skips authors already collected (seen_author_ids) and only
    fetches up to `need_count` new authors.

    Returns (profiles, n_qualified, total_papers):
      - profiles: newly-built profile dicts (excluding already-seen authors)
      - n_qualified: total number of qualified authors found this round
      - total_papers: number of papers pulled from search this round
    """
    # ── Phase 1: Search & aggregate authors ────────────────────────
    author_papers: dict[str, list[dict]] = {}
    total_papers = 0

    for query in cfg["search_queries"]:
        papers = client.search_papers(query, limit=papers_per_query)
        total_papers += len(papers)
        for p in papers:
            for a in p["authors"]:
                aid = a["author_id"]
                if not aid or aid in seen_author_ids:
                    continue
                if aid not in author_papers:
                    author_papers[aid] = []
                if len(author_papers[aid]) < max_pubs * 3:
                    author_papers[aid].append(p)

    # Rank qualified authors (skip already-seen)
    ranked = sorted(
        [(aid, ps) for aid, ps in author_papers.items()
         if aid and aid != "None" and len(ps) >= min_papers],
        key=lambda x: len(x[1]), reverse=True,
    )

    # ── Phase 2: Fetch complete papers for top-N authors ───────────
    top = ranked[:need_count]
    enriched: dict[str, list[dict]] = {}

    for aid, ps in top:
        s2_papers = client.get_author_papers(aid, limit=max_pubs * 3)
        if not s2_papers:
            s2_papers = ps
        existing_ids = {p["s2_paper_id"] for p in s2_papers}
        for p in ps:
            if p["s2_paper_id"] not in existing_ids:
                s2_papers.append(p)
        enriched[aid] = s2_papers

    # ── Phase 3: Build profiles ────────────────────────────────────
    profiles = build_profiles(enriched, area_name, max_pubs, anonymizer)

    # Mark these authors as collected (dedup across rounds)
    for aid in enriched:
        seen_author_ids.add(aid)

    return profiles, len(ranked), total_papers


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Scholar profile collection"
    )
    parser.add_argument("--areas", nargs="+", default=["cs.CV"])
    parser.add_argument("--num-authors", type=int, default=20)
    parser.add_argument("--min-papers", type=int, default=3)
    parser.add_argument("--max-pubs", type=int, default=10)
    parser.add_argument("--papers-per-query", type=int, default=50)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--output", type=str,
                       default="config/profiles/profiles_140.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("S2_API_KEY", "")
    if not api_key and not args.dry_run:
        logger.error(
            "Set S2_API_KEY env var or pass --api-key. "
            "Get a free key at semanticscholar.org/product/api"
        )
        return

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.dry_run:
        logger.info("=== DRY RUN (Semantic Scholar) ===")
        logger.info("Areas: %s", args.areas)
        for a in args.areas:
            if a in TARGET_AREAS:
                logger.info("  %s: %d queries", a,
                           len(TARGET_AREAS[a]["search_queries"]))
        logger.info("API key: %s", "SET" if api_key else "NOT SET")
        logger.info("Rate limit: 1 req/sec")
        return

    client = S2Client(api_key)
    anonymizer = Anonymizer()
    all_profiles = []

    for area_name in args.areas:
        if area_name not in TARGET_AREAS:
            continue

        cfg = TARGET_AREAS[area_name]
        logger.info("=" * 60)
        logger.info("Area: %s  |  Queries: %d  |  1 req/sec",
                   area_name, len(cfg["search_queries"]))
        logger.info("=" * 60)

        seen_author_ids = set()
        area_profiles = []

        # ── Re-collection strategies (in order) ────────────────────
        # First widen the search (more papers per query), then lower the
        # qualification threshold (min_papers), until we reach num_authors
        # or exhaust all strategies.
        strategies = [
            (args.min_papers, args.papers_per_query, "默认参数"),
            (args.min_papers, args.papers_per_query * 2, "扩大搜索×2"),
            (args.min_papers, args.papers_per_query * 4, "扩大搜索×4"),
            (max(2, args.min_papers - 1), args.papers_per_query * 2, "降低门槛"),
            (1, args.papers_per_query * 2, "最低门槛"),
        ]

        for round_i, (min_papers, ppq, desc) in enumerate(strategies):
            if len(area_profiles) >= args.num_authors:
                break

            need = args.num_authors - len(area_profiles)
            logger.info("补采轮 %d: %s (min_papers=%d, papers_per_query=%d, 还需 %d)",
                       round_i + 1, desc, min_papers, ppq, need)

            profiles, n_qualified, total_papers = collect_area_round(
                client, area_name, cfg, min_papers, ppq,
                args.max_pubs, anonymizer, seen_author_ids, need,
            )
            area_profiles.extend(profiles)

            logger.info("  → 本轮采到 %d, 累计 %d/%d (合格作者 %d, 论文 %d)",
                       len(profiles), len(area_profiles), args.num_authors,
                       n_qualified, total_papers)

            for p in profiles:
                n_abs = sum(
                    1 for pub in p["publications"]
                    if pub["abstract"] and len(pub["abstract"]) > 50
                )
                logger.info(
                    "    %s | %d pubs (%d w/abstract) | kw: %s",
                    p["profile_id"], len(p["publications"]), n_abs,
                    ", ".join(p["sub_areas"][:3]),
                )

            # If no more qualified authors even with relaxed params, stop.
            if n_qualified == 0:
                logger.info("  无更多合格作者, 提前结束补采")
                break

        logger.info("  → %s 领域共 %d profiles\n", area_name, len(area_profiles))
        all_profiles.extend(area_profiles)

        # Save
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_profiles, f, indent=2, ensure_ascii=False)

    # ── Report ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("COLLECTION COMPLETE — %d profiles", len(all_profiles))

    if not all_profiles:
        return

    total_pubs = sum(len(p["publications"]) for p in all_profiles)
    all_titles = set()
    total_abs = 0
    total_chars = 0
    for p in all_profiles:
        for pub in p["publications"]:
            all_titles.add(pub["title"].lower())
            if pub["abstract"] and len(pub["abstract"]) > 50:
                total_abs += 1
                total_chars += len(pub["abstract"])

    logger.info("Pub slots: %d | Unique: %d (%.0f%%)",
               total_pubs, len(all_titles),
               100 * len(all_titles) / max(1, total_pubs))
    logger.info("With abstracts: %d/%d (%.0f%%)",
               total_abs, total_pubs,
               100 * total_abs / max(1, total_pubs))
    if total_abs > 0:
        logger.info("Avg abstract: %.0f chars", total_chars / total_abs)

    for p in all_profiles[:3]:
        logger.info("\n%s | %d pubs", p["profile_id"], len(p["publications"]))
        for pub in p["publications"][:2]:
            logger.info("  - %s", pub["title"][:100])
            if pub["abstract"]:
                logger.info("    %s...", pub["abstract"][:120])

    logger.info("\nOutput: %s", output_path.absolute())


if __name__ == "__main__":
    main()
