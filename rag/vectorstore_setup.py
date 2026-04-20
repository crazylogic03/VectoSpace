from __future__ import annotations
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_STORE = _ROOT / "model_artifacts"
VECTORSTORE_PATH = Path(os.getenv("VECTORSTORE_PATH", str(_DEFAULT_STORE)))
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

_INDEX_FILE = VECTORSTORE_PATH / "faiss_index.pkl"
_DOCS_FILE  = VECTORSTORE_PATH / "documents.json"

RESOURCE_CORPUS: list[dict[str, Any]] = [
    {
        "title":   "Khan Academy – Mathematics",
        "url":     "https://www.khanacademy.org/math",
        "domain":  ["Mathematics", "math"],
        "summary": "Free, self-paced video lessons and practice exercises covering arithmetic, algebra, geometry, trigonometry, and calculus.",
    },
    {
        "title":   "Paul's Online Math Notes",
        "url":     "https://tutorial.math.lamar.edu/",
        "domain":  ["Mathematics", "math", "calculus", "algebra"],
        "summary": "Comprehensive, exam-friendly notes and worked examples for Algebra, Calculus I–III, Differential Equations, and Linear Algebra.",
    },
    {
        "title":   "Brilliant – Math Courses",
        "url":     "https://brilliant.org/courses/math/",
        "domain":  ["Mathematics", "math", "problem solving"],
        "summary": "Interactive problem-solving courses in mathematics and logic teaching reasoning rather than rote memorisation.",
    },
    {
        "title":   "MIT OpenCourseWare – Mathematics",
        "url":     "https://ocw.mit.edu/courses/mathematics/",
        "domain":  ["Mathematics", "math", "linear algebra", "calculus"],
        "summary": "Free MIT lecture notes, assignments, and exams for undergraduate mathematics courses.",
    },
    {
        "title":   "Wolfram MathWorld",
        "url":     "https://mathworld.wolfram.com/",
        "domain":  ["Mathematics", "math", "reference"],
        "summary": "Extensive mathematics encyclopedia covering definitions, theorems, and examples across all branches of mathematics.",
    },
    {
        "title":   "Khan Academy – Science",
        "url":     "https://www.khanacademy.org/science",
        "domain":  ["Science", "physics", "chemistry", "biology"],
        "summary": "Free video lessons and practice quizzes covering biology, chemistry, physics, and Earth sciences.",
    },
    {
        "title":   "CK-12 – Science Flexbooks",
        "url":     "https://www.ck12.org/student/",
        "domain":  ["Science", "physics", "chemistry", "biology"],
        "summary": "Open-source digital textbooks with simulations and adaptive practice for physics, chemistry, biology, and earth science.",
    },
    {
        "title":   "PhET Interactive Science Simulations",
        "url":     "https://phet.colorado.edu/",
        "domain":  ["Science", "physics", "chemistry"],
        "summary": "Free, research-based science simulations for experimenting with physics, chemistry, and earth science concepts.",
    },
    {
        "title":   "NCBI – Biology and Life Sciences Resources",
        "url":     "https://www.ncbi.nlm.nih.gov/education/",
        "domain":  ["Science", "biology", "life sciences"],
        "summary": "Educational resources including tutorials on genetics, genomics, and molecular biology.",
    },
    {
        "title":   "British Council – LearnEnglish",
        "url":     "https://learnenglish.britishcouncil.org/",
        "domain":  ["English", "language", "grammar", "writing"],
        "summary": "Free English learning resources: grammar, listening, reading passages, and writing guides.",
    },
    {
        "title":   "Purdue OWL – Writing and Grammar",
        "url":     "https://owl.purdue.edu/owl/general_writing/",
        "domain":  ["English", "writing", "grammar", "academic writing"],
        "summary": "Definitive reference for academic writing, grammar rules, citation formats, and essay structure.",
    },
    {
        "title":   "BBC Learning English",
        "url":     "https://www.bbc.co.uk/learningenglish/",
        "domain":  ["English", "language", "vocabulary", "listening"],
        "summary": "Daily grammar tips, vocabulary lessons, and real-world English through audio and video content.",
    },
    {
        "title":   "Project Gutenberg – Free Classic Literature",
        "url":     "https://www.gutenberg.org/",
        "domain":  ["English", "literature", "reading"],
        "summary": "Over 70,000 free eBooks of classic literature to improve reading comprehension and analytical skills.",
    },
    {
        "title":   "Coursera – Learning How to Learn",
        "url":     "https://www.coursera.org/learn/learning-how-to-learn",
        "domain":  ["Attendance", "Study Time", "habits", "productivity"],
        "summary": "Science-backed MOOC covering memory, procrastination, and effective study techniques.",
    },
    {
        "title":   "Todoist – Student Planning Guide",
        "url":     "https://todoist.com/productivity-methods/student-study-plan",
        "domain":  ["Study Time", "Attendance", "planning", "organisation"],
        "summary": "A practical guide to building a structured weekly study timetable and maintaining academic momentum.",
    },
    {
        "title":   "Forest App – Focus & Study Timer",
        "url":     "https://www.forestapp.cc/",
        "domain":  ["Study Time", "focus", "productivity"],
        "summary": "Gamified Pomodoro timer app that helps students stay focused and builds sustained study habits.",
    },
    {
        "title":   "Cal Newport – Study Strategies (Blog)",
        "url":     "https://calnewport.com/blog/",
        "domain":  ["Study Time", "Study habits", "deep work"],
        "summary": "Research-driven blog on deliberate practice, deep work, and effective study strategies.",
    },
    {
        "title":   "Anki – Spaced Repetition Flashcards",
        "url":     "https://apps.ankiweb.net/",
        "domain":  ["Mathematics", "Science", "English", "general", "memory"],
        "summary": "Free flashcard software using spaced-repetition algorithms to optimise long-term memorisation.",
    },
    {
        "title":   "OpenStax – Free Peer-reviewed Textbooks",
        "url":     "https://openstax.org/subjects",
        "domain":  ["Mathematics", "Science", "English", "general"],
        "summary": "High-quality, peer-reviewed, openly-licensed textbooks in maths, sciences, and humanities.",
    },
    {
        "title":   "Quizlet – Flashcards & Study Sets",
        "url":     "https://quizlet.com/",
        "domain":  ["Mathematics", "Science", "English", "general", "revision"],
        "summary": "Millions of student-created study sets with multiple learning modes for vocabulary and revision.",
    },
    {
        "title":   "edX – University Online Courses",
        "url":     "https://www.edx.org/",
        "domain":  ["Mathematics", "Science", "English", "general"],
        "summary": "Online courses from top universities (MIT, Harvard, etc.) accessible to all for audit.",
    },
]

def _try_import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer
    except ImportError:
        return None

def _build_tfidf_embedder(texts: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    mat = vec.fit_transform(texts).toarray()
    return mat, vec

def _build_faiss_index(embeddings):
    try:
        import faiss  # type: ignore
        import numpy as np
        mat = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(mat)
        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(mat)
        return ("faiss", index)
    except ImportError:
        import numpy as np
        mat = np.array(embeddings, dtype="float32")
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
        mat   = mat / norms
        return ("numpy", mat)

def build_vectorstore(force: bool = False) -> dict:
    VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
    if not force and _INDEX_FILE.exists() and _DOCS_FILE.exists():
        try:
            with open(_INDEX_FILE, "rb") as fh:
                store = pickle.load(fh)
            with open(_DOCS_FILE, "r", encoding="utf-8") as fh:
                store["documents"] = json.load(fh)
            return store
        except:
            pass
    embed_texts = [f"{doc['title']}. {' '.join(doc['domain'])}. {doc['summary']}" for doc in RESOURCE_CORPUS]
    SentenceTransformer = _try_import_sentence_transformers()
    if SentenceTransformer is not None:
        model      = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = model.encode(embed_texts, show_progress_bar=False, normalize_embeddings=True)
        embed_type = "sentence_transformers"
        embedder   = model
    else:
        embeddings, embedder = _build_tfidf_embedder(embed_texts)
        embed_type = "tfidf"
    index_type, index = _build_faiss_index(embeddings)
    store = {
        "index_type": index_type if embed_type == "sentence_transformers" else "tfidf",
        "index":      index,
        "documents":  RESOURCE_CORPUS,
        "embedder":   embedder,
        "embed_type": embed_type,
    }
    try:
        with open(_INDEX_FILE, "wb") as fh:
            pickle.dump({
                "index_type": store["index_type"],
                "index":      store["index"],
                "embedder":   store["embedder"],
                "embed_type": store["embed_type"],
            }, fh, protocol=pickle.HIGHEST_PROTOCOL)
        with open(_DOCS_FILE, "w", encoding="utf-8") as fh:
            json.dump(RESOURCE_CORPUS, fh, indent=2, ensure_ascii=False)
    except:
        pass
    return store

def search_vectorstore(query: str, store: dict, top_k: int = 5, min_score: float = 0.10) -> list[dict]:
    import numpy as np
    documents  = store["documents"]
    embedder   = store["embedder"]
    index_type = store["index_type"]
    embed_type = store["embed_type"]
    if embed_type == "sentence_transformers":
        q_vec = embedder.encode([query], normalize_embeddings=True)
    else:
        vectoriser = embedder
        q_vec = vectoriser.transform([query]).toarray()
        norm  = np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-10
        q_vec = q_vec / norm
    q_vec = np.array(q_vec, dtype="float32")
    if index_type == "faiss":
        scores, indices = store["index"].search(q_vec, min(top_k, len(documents)))
        scores  = scores[0].tolist()
        indices = indices[0].tolist()
    else:
        corpus_mat = store["index"]
        raw_scores = corpus_mat.dot(q_vec[0])
        top_idx    = np.argsort(raw_scores)[::-1][:top_k]
        indices    = top_idx.tolist()
        scores     = raw_scores[top_idx].tolist()
    results = []
    for idx, score in zip(indices, scores):
        if idx < 0 or idx >= len(documents) or float(score) < min_score:
            continue
        doc = dict(documents[idx])
        doc["score"] = round(float(score), 4)
        results.append(doc)
    return results

if __name__ == "__main__":
    store = build_vectorstore(force=True)
    test_queries = ["Mathematics score is critically low", "Student has poor English writing"]
    for q in test_queries:
        hits = search_vectorstore(q, store, top_k=3)
        for h in hits:
            print(f"[{h['score']:.3f}] {h['title']} -> {h['url']}")
