"""
RAG — QA Engine
Retrieves context and generates a conversational answer.

Answer-generation rules:
  - With LLM key: build a grounded prompt → call LLM → return its response.
  - Without LLM key: synthesize a natural-language answer directly from the
    retrieved chunks without exposing raw RAG data to the user.
  - Never dump raw chunk text, index numbers [1][2], "Source:" labels,
    or unprocessed file lists to the user.
  - When context is insufficient: say so clearly in one sentence.
"""

import logging
import re
from typing import Dict, List, Optional

from config import LLM_PROVIDER, LLM_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, TOP_K_RETRIEVAL
from rag.retriever import VectorRetriever

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = max(TOP_K_RETRIEVAL, 6)


class QAEngine:
    """
    Retrieval-Augmented Generation engine.

    Pipeline:
      1. Auto-load index from disk if empty.
      2. Retrieve top-k relevant chunks.
      3a. With LLM: build prompt → call LLM → return response.
      3b. Without LLM: synthesize a natural-language answer from chunks.
      In both cases the user sees ONLY the final human-readable answer.
    """

    def __init__(self, retriever: Optional[VectorRetriever] = None):
        self.retriever = retriever or VectorRetriever()
        self._llm_client = None

    # ── Index management ─────────────────────────────────────

    def build_index(self, shared_knowledge) -> None:
        from rag.embeddings import embed_chunks
        from rag.document_processor import process_all_repos
        chunks = process_all_repos(shared_knowledge)
        if not chunks:
            logger.warning("No chunks produced — index not built.")
            return
        chunks = embed_chunks(chunks)
        self.retriever.build(chunks)
        self.retriever.save()
        logger.info("RAG index built: %d chunks.", len(chunks))

    def ensure_index(self, shared_knowledge=None) -> None:
        if self.retriever.chunks:
            return
        loaded = self.retriever.load()
        if loaded and self.retriever.chunks:
            logger.info("RAG index loaded from disk (%d chunks).", len(self.retriever.chunks))
            return
        if shared_knowledge is not None:
            shared_knowledge.load_from_disk()
            if shared_knowledge.all_repos():
                logger.info(
                    "Rebuilding RAG index from disk knowledge (%d repos).",
                    len(shared_knowledge.all_repos()),
                )
                self.build_index(shared_knowledge)

    def has_knowledge(self) -> bool:
        return bool(self.retriever.chunks)

    # ── LLM client ───────────────────────────────────────────

    def _get_llm_client(self):
        if self._llm_client:
            return self._llm_client
        if not LLM_API_KEY:
            return None
        try:
            if LLM_PROVIDER in ("openai", "groq"):
                from openai import OpenAI
                kwargs = {"api_key": LLM_API_KEY}
                if LLM_PROVIDER == "groq":
                    kwargs["base_url"] = "https://api.groq.com/openai/v1"
                self._llm_client = OpenAI(**kwargs)
        except ImportError as exc:
            logger.warning("LLM client import failed: %s", exc)
        return self._llm_client

    # ── LLM prompt ───────────────────────────────────────────

    @staticmethod
    def _build_prompt(question: str, context_chunks: List[Dict]) -> str:
        """
        Build a grounded prompt for the LLM.
        Context is presented as plain paragraphs — no [1][2] index numbers
        so they cannot leak into the generated answer.
        """
        context_parts = []
        for chunk in context_chunks:
            text = chunk.get("text", "").strip()
            if text:
                context_parts.append(text)

        context_str = "\n\n".join(context_parts)

        return (
            "You are CodeLens AI, a helpful software assistant that answers "
            "developer questions about analyzed GitHub repositories.\n\n"
            "Instructions:\n"
            "- Answer the question in clear, natural, conversational English.\n"
            "- Use complete sentences. Do NOT use bullet dumps of raw file paths "
            "or class names without explanation.\n"
            "- Use ONLY information present in the repository context below. "
            "Do NOT invent file names, class names, or methods.\n"
            "- When mentioning a file or class, name it naturally within the sentence "
            "(e.g. 'The main entry point is `app.py`').\n"
            "- If the context does not contain enough information, say: "
            "'I couldn't find enough information in the analyzed repository to answer "
            "that confidently.'\n"
            "- Do NOT include labels like 'Source:', '[1]', or 'Retrieved context:'.\n"
            "- Keep the answer focused and under 200 words unless detail is essential.\n\n"
            f"=== Repository Context ===\n{context_str}\n\n"
            f"=== Question ===\n{question}\n\n"
            "=== Answer ==="
        )

    # ── No-LLM synthesis ─────────────────────────────────────

    @classmethod
    def _synthesize_answer(cls, question: str, chunks: List[Dict]) -> str:
        """
        Produce a natural-language answer from retrieved chunks
        without calling an LLM and without exposing raw chunk data.

        Extracts structured facts (layers, entry points, key components,
        statistics, etc.) from the chunk text and composes them into
        coherent sentences matched to the question type.
        """
        if not chunks:
            return ("I couldn't find enough information in the analyzed repository "
                    "to answer that confidently.")

        # Combine all chunk text for extraction
        all_text = "\n".join(c.get("text", "") for c in chunks)
        repo_name = chunks[0].get("source", "the repository")
        q = question.lower().strip().rstrip("?")

        # ── Extract structured facts ──────────────────────────
        facts = cls._extract_facts(all_text)

        # ── Route to the right answer builder ────────────────
        if any(kw in q for kw in ("entry point", "main file", "start", "starts from",
                                   "starting point", "how to run", "runs from")):
            return cls._build_entry_point_answer(facts, repo_name)

        if any(kw in q for kw in ("auth", "login", "password", "token", "oauth",
                                   "jwt", "permission", "security", "credential")):
            return cls._build_topic_answer(q, facts, repo_name,
                                           ["auth", "login", "token", "oauth", "jwt",
                                            "password", "permission", "credential"],
                                           "authentication")

        if any(kw in q for kw in ("database", "sql", "model", "orm", "storage",
                                   "persist", "query", "migration", "schema")):
            return cls._build_topic_answer(q, facts, repo_name,
                                           ["database", "db", "sql", "model", "orm",
                                            "migration", "schema", "query"],
                                           "data storage")

        if any(kw in q for kw in ("component", "architecture", "structure", "major",
                                   "layer", "module", "overview", "organized",
                                   "connected", "relationship")):
            return cls._build_architecture_answer(facts, repo_name)

        if any(kw in q for kw in ("statistic", "how many", "count", "total",
                                   "file", "class", "method", "function")):
            return cls._build_statistics_answer(facts, repo_name, q)

        # General: compose from whatever facts we have
        return cls._build_general_answer(q, facts, repo_name)

    # ── Fact extractor ────────────────────────────────────────

    @staticmethod
    def _extract_facts(text: str) -> Dict:
        """
        Parse the combined chunk text into a structured dict of facts.
        Only extracts lines that follow the one-concept-per-line format
        produced by SharedKnowledge.flat_text_for_rag().
        """
        facts: Dict = {
            "repo":         "",
            "description":  "",
            "primary_lang": "",
            "languages":    [],
            "statistics":   "",
            "dependencies": [],
            "entry_points": [],
            "layers":       {},        # layer_name -> [file, ...]
            "key_components": [],
            "files":        [],
            "classes":      [],
            "methods":      [],
            "functions":    [],
            "imports":      [],
            "summary":      "",
        }

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            ll = line.lower()

            if ll.startswith("repository:"):
                facts["repo"] = line.split(":", 1)[1].strip()

            elif ll.startswith("description:"):
                facts["description"] = line.split(":", 1)[1].strip()

            elif ll.startswith("primary language:"):
                facts["primary_lang"] = line.split(":", 1)[1].strip()

            elif ll.startswith("languages used:"):
                facts["languages"] = [l.strip() for l in
                                      line.split(":", 1)[1].split(",")]

            elif ll.startswith("statistics:"):
                facts["statistics"] = line.split(":", 1)[1].strip()

            elif ll.startswith("detected dependencies:"):
                facts["dependencies"] = [d.strip() for d in
                                         line.split(":", 1)[1].split(",")][:12]

            elif ll.startswith("entry point file:"):
                ep = line.split(":", 1)[1].strip()
                # show only the filename, not the full path
                ep_short = ep.replace("\\", "/").split("/")[-1]
                if ep_short not in facts["entry_points"]:
                    facts["entry_points"].append(ep_short)

            elif " layer files:" in ll:
                layer_name = line.split(" layer files:")[0].strip()
                layer_files = [f.strip().replace("\\", "/").split("/")[-1]
                               for f in line.split(":", 1)[1].split(",")]
                facts["layers"][layer_name] = layer_files[:5]

            elif ll.startswith("key component:"):
                rest = line.split(":", 1)[1].strip()
                # Format: "label (type: File, connectivity degree: N)"
                comp_name = rest.split("(")[0].strip()
                comp_name = comp_name.replace("\\", "/").split("/")[-1]
                if comp_name and comp_name not in facts["key_components"]:
                    facts["key_components"].append(comp_name)

            elif ll.startswith("file:"):
                rest = line.split(":", 1)[1].strip()
                fname = rest.split("(")[0].strip()
                fname = fname.replace("\\", "/").split("/")[-1]
                if fname and fname not in facts["files"]:
                    facts["files"].append(fname)

            elif ll.startswith("class ") and " in file " in ll:
                class_name = line.split()[1]
                if class_name not in facts["classes"]:
                    facts["classes"].append(class_name)

            elif ll.startswith("function ") and " defined in file " in ll:
                func_name = line.split()[1]
                if func_name not in facts["functions"]:
                    facts["functions"].append(func_name)

            elif "--- architecture" in ll or "--- architecture summary ---" in ll:
                continue  # section header

            elif ll.startswith("architecture analysis"):
                facts["summary"] = line

        return facts

    # ── Answer builders ───────────────────────────────────────

    @staticmethod
    def _build_entry_point_answer(facts: Dict, repo: str) -> str:
        eps = facts["entry_points"]
        if not eps:
            return ("The main entry point was not explicitly identified in the "
                    f"analyzed repository knowledge for {repo}.")
        if len(eps) == 1:
            return (f"The main entry point of {repo} is `{eps[0]}`. "
                    "This is where the application starts execution.")
        listed = ", ".join(f"`{e}`" for e in eps[:4])
        return (f"The repository has multiple entry points. The primary ones are "
                f"{listed}. These files are where the application initializes and "
                "starts execution.")

    @staticmethod
    def _build_topic_answer(question: str, facts: Dict,
                             repo: str, keywords: List[str], topic: str) -> str:
        # Look for relevant classes, files, functions matching the keywords
        relevant_classes = [c for c in facts["classes"]
                            if any(kw in c.lower() for kw in keywords)]
        relevant_files   = [f for f in facts["files"]
                            if any(kw in f.lower() for kw in keywords)]
        relevant_funcs   = [f for f in facts["functions"]
                            if any(kw in f.lower() for kw in keywords)]
        deps             = [d for d in facts["dependencies"]
                            if any(kw in d.lower() for kw in keywords)]

        parts = []
        if relevant_classes:
            cls_list = ", ".join(f"`{c}`" for c in relevant_classes[:4])
            parts.append(f"The main {topic} classes are {cls_list}.")
        if relevant_files:
            f_list = ", ".join(f"`{f}`" for f in relevant_files[:4])
            parts.append(f"Relevant files include {f_list}.")
        if relevant_funcs:
            fn_list = ", ".join(f"`{f}`" for f in relevant_funcs[:4])
            parts.append(f"Key functions involved are {fn_list}.")
        if deps:
            d_list = ", ".join(deps[:4])
            parts.append(f"External libraries used include {d_list}.")

        if parts:
            return (f"Based on the analyzed repository, here is what was found "
                    f"regarding {topic}:\n\n" + "  ".join(parts))
        return (f"I couldn't find enough information about {topic} in the "
                "analyzed repository to answer that confidently.")

    @staticmethod
    def _build_architecture_answer(facts: Dict, repo: str) -> str:
        parts = []

        # Description
        if facts["description"]:
            parts.append(facts["description"])

        # Language
        if facts["primary_lang"]:
            parts.append(f"It is primarily written in {facts['primary_lang']}.")

        # Layers
        if facts["layers"]:
            layer_descs = {
                "Presentation": "handles user-facing features such as APIs, routes, and views",
                "Service":      "contains the core business logic and processing",
                "Data":         "manages data models, database access, and persistence",
                "Unclassified": "contains supporting utilities and helper modules",
            }
            layer_sentences = []
            for lname, lfiles in facts["layers"].items():
                desc = layer_descs.get(lname,
                       f"contains {lname.lower()}-related code")
                layer_sentences.append(
                    f"The **{lname} layer** {desc}"
                    + (f" (e.g. `{lfiles[0]}`)" if lfiles else "") + "."
                )
            parts.append(
                "The project is organised into "
                + str(len(facts["layers"]))
                + " architectural layers: "
                + " ".join(layer_sentences)
            )

        # Entry points
        if facts["entry_points"]:
            eps = ", ".join(f"`{e}`" for e in facts["entry_points"][:3])
            parts.append(f"The main entry point(s) are {eps}.")

        # Key components
        if facts["key_components"]:
            comps = ", ".join(f"`{c}`"
                              for c in facts["key_components"][:5]
                              if not c.startswith("__"))
            if comps:
                parts.append(f"The most central components are {comps}.")

        # Dependencies
        if facts["dependencies"]:
            deps = ", ".join(facts["dependencies"][:6])
            parts.append(f"Key external dependencies include {deps}.")

        # Statistics
        if facts["statistics"]:
            parts.append(facts["statistics"])

        if not parts:
            return ("I couldn't find enough information in the analyzed repository "
                    "to describe its architecture confidently.")

        return "\n\n".join(parts)

    @staticmethod
    def _build_statistics_answer(facts: Dict, repo: str, question: str) -> str:
        if facts["statistics"]:
            return (f"Here are the statistics for the analyzed repository:\n\n"
                    f"{facts['statistics']}")
        counts = []
        if facts["files"]:
            counts.append(f"{len(facts['files'])} files")
        if facts["classes"]:
            counts.append(f"{len(facts['classes'])} classes")
        if facts["functions"]:
            counts.append(f"{len(facts['functions'])} functions")
        if counts:
            return (f"The analyzed repository contains at least "
                    + ", ".join(counts) + " in the retrieved context.")
        return ("Detailed statistics were not found in the analyzed "
                "repository knowledge.")

    @staticmethod
    def _build_general_answer(question: str, facts: Dict, repo: str) -> str:
        """Compose a general answer when no specific pattern matches."""
        # Try to find something relevant in the facts
        q_words = set(re.sub(r'[^a-z0-9 ]', '', question).split())
        q_words -= {"the", "a", "an", "is", "are", "was", "were", "what",
                    "how", "why", "when", "which", "does", "do", "in", "of",
                    "for", "to", "at", "by", "with", "this", "that"}

        candidates = (
            facts["classes"] + facts["functions"] + facts["files"] +
            facts["key_components"] + facts["entry_points"]
        )
        relevant = [c for c in candidates
                    if any(w in c.lower() for w in q_words if len(w) > 3)]

        if relevant:
            listed = ", ".join(f"`{c}`" for c in relevant[:6])
            return (f"Based on the repository analysis, the most relevant items "
                    f"found are: {listed}.")

        # Last resort: give architecture overview
        if facts["layers"] or facts["description"]:
            return QAEngine._build_architecture_answer(facts, repo)

        return ("I couldn't find enough information in the analyzed repository "
                "to answer that confidently.")

    # ── Main query ────────────────────────────────────────────

    def query(self, question: str, top_k: int = _DEFAULT_TOP_K) -> Dict:
        """
        Answer any natural-language developer question.

        Returns:
            {
                "answer"       : str   — final human-readable answer only
                "sources"      : [str] — repo names used (not shown to user by default)
                "chunks_used"  : int
                "llm_used"     : bool
                "has_knowledge": bool
            }
        """
        if not self.has_knowledge():
            return {
                "answer": (
                    "No repository knowledge is available yet. "
                    "Please run the analysis pipeline first: go to the home page, "
                    "enter repository URLs, click 'Find & Analyze', then click "
                    "'Analyze' on a repository in the Discovery results."
                ),
                "sources":       [],
                "chunks_used":   0,
                "llm_used":      False,
                "has_knowledge": False,
            }

        retrieved = self.retriever.retrieve(question, top_k=top_k)

        if not retrieved:
            return {
                "answer": (
                    "I couldn't find enough information in the analyzed repository "
                    "to answer that confidently. The repository may not contain "
                    "code related to your question, or it may not have been "
                    "fully analyzed yet."
                ),
                "sources":       [],
                "chunks_used":   0,
                "llm_used":      False,
                "has_knowledge": True,
            }

        sources = list({c.get("source", "") for c in retrieved if c.get("source")})

        # ── Try LLM first ─────────────────────────────────────
        client = self._get_llm_client()
        if client is not None:
            prompt = self._build_prompt(question, retrieved)
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=LLM_MAX_TOKENS,
                    temperature=0.1,
                )
                answer = response.choices[0].message.content.strip()
                return {
                    "answer":        answer,
                    "sources":       sources,
                    "chunks_used":   len(retrieved),
                    "llm_used":      True,
                    "has_knowledge": True,
                }
            except Exception as exc:
                logger.error("LLM call failed: %s", exc)
                # Fall through to no-LLM path silently

        # ── No-LLM synthesis (or LLM failed) ─────────────────
        answer = self._synthesize_answer(question, retrieved)
        return {
            "answer":        answer,
            "sources":       sources,
            "chunks_used":   len(retrieved),
            "llm_used":      False,
            "has_knowledge": True,
        }
