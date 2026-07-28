"""Map-first answers for the EduBridge chat.

The map's school records are the primary source.  A web lookup is only used
when the requested school, district, county, or metric is not in the map.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from statistics import mean
from typing import Any, Optional
from urllib.request import Request, urlopen


_SCHOOLS: Optional[list[dict[str, Any]]] = None


def _schools() -> list[dict[str, Any]]:
    global _SCHOOLS
    if _SCHOOLS is None:
        map_html = Path(__file__).with_name("map.html").read_text(encoding="utf-8")
        match = re.search(r"const SCHOOLS = (\[.*?\]);\s*\n", map_html, re.DOTALL)
        if not match:
            raise RuntimeError("Could not load the school records from map.html.")
        _SCHOOLS = json.loads(match.group(1))
    return _SCHOOLS


def _normal(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _record_matches(question: str, record: dict[str, Any]) -> bool:
    query = _normal(question)
    fields = ("name", "district", "county", "address")
    for field in fields:
        value = _normal(record.get(field))
        if len(value) >= 4 and value in query:
            return True
    return False


def _metric(question: str) -> tuple[str, str] | None:
    query = _normal(question)
    metrics = (
        (("graduation", "graduate", "grad rate"), "grad_rate_pct", "graduation rate"),
        (("absence", "absentee", "attendance"), "absence_rate_pct", "absence rate"),
        (("student teacher", "teacher ratio", "class size"), "student_teacher_ratio", "student-to-teacher ratio"),
        (("poverty", "free lunch", "frl"), "frl_rate", "free/reduced lunch rate"),
        (("spg", "performance", "score", "grade"), "spg_score", "school performance score"),
        (("enrollment", "students"), "enrollment", "enrollment"),
    )
    for terms, field, label in metrics:
        if any(term in query for term in terms):
            return field, label
    return None


def _format_value(field: str, value: float) -> str:
    if field == "grad_rate_pct":
        return f"{value / 100:.1f}%"
    if field in {"absence_rate_pct", "frl_rate"}:
        return f"{value:.1f}%"
    if field == "student_teacher_ratio":
        return f"{value:.1f}:1"
    if field == "enrollment":
        return f"{round(value):,}"
    return f"{value:.1f}"


def _map_answer(question: str) -> Optional[dict[str, str]]:
    records = [record for record in _schools() if _record_matches(question, record)]
    if not records:
        return None

    metric = _metric(question)
    exact_school = next(
        (record for record in records if _normal(record.get("name")) in _normal(question)),
        None,
    )

    if exact_school:
        if metric and exact_school.get(metric[0]) is not None:
            return {
                "answer": (
                    f"According to the map, {exact_school['name']} in {exact_school['county']} County has a "
                    f"{metric[1]} of {_format_value(metric[0], float(exact_school[metric[0]]))}."
                ),
                "source": "map",
            }
        facts = []
        if exact_school.get("spg_score") is not None:
            facts.append(f"school performance score {exact_school['spg_score']}")
        if exact_school.get("absence_rate_pct") is not None:
            facts.append(f"absence rate {exact_school['absence_rate_pct']}%")
        if exact_school.get("enrollment") is not None:
            facts.append(f"enrollment {exact_school['enrollment']:,}")
        if facts:
            return {
                "answer": f"According to the map, {exact_school['name']} ({exact_school['county']} County) has " + ", ".join(facts) + ".",
                "source": "map",
            }

    if metric:
        field, label = metric
        values = [float(record[field]) for record in records if record.get(field) is not None]
        if values:
            place = records[0].get("county") or records[0].get("district") or "the matched area"
            return {
                "answer": (
                    f"Using {len(values)} mapped school record(s) in {place}, the average {label} is "
                    f"{_format_value(field, mean(values))}."
                ),
                "source": "map",
            }

    place = records[0].get("county") or records[0].get("district") or "that area"
    return {
        "answer": f"The map has {len(records)} school record(s) for {place}. Ask about performance, attendance, graduation, enrollment, or student-to-teacher ratios for a map-based comparison.",
        "source": "map",
    }


def _web_answer(question: str) -> Optional[dict[str, Any]]:
    """Use Gemini's Google Search grounding when the local map has no answer."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    prompt = (
        "Answer this question about North Carolina education clearly and concisely. "
        "Use Google Search for current, trustworthy information. Do not invent facts. "
        "Question: " + question
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
    }
    request = Request(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=" + api_key,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=8) as response:
            result = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    candidates = result.get("candidates") or []
    if not candidates:
        return None
    candidate = candidates[0]
    parts = candidate.get("content", {}).get("parts", [])
    answer = "".join(part.get("text", "") for part in parts).strip()
    chunks = candidate.get("groundingMetadata", {}).get("groundingChunks", [])
    sources = []
    for chunk in chunks:
        web = chunk.get("web")
        if web and web.get("uri"):
            sources.append({"title": web.get("title") or web["uri"], "url": web["uri"]})
    if answer:
        return {"answer": answer, "source": "web", "sources": sources[:4]}
    return None


def answer_map_question(question: str) -> dict[str, Any]:
    question = question.strip()
    if not question:
        return {"answer": "Ask about a North Carolina school, county, district, or education metric.", "source": "map"}

    map_result = _map_answer(question)
    if map_result:
        return map_result

    web_result = _web_answer(question)
    if web_result:
        return web_result

    return {
        "answer": "I could not find that in the map or retrieve a reliable web result. Try naming a North Carolina school, county, district, or metric.",
        "source": "unavailable",
    }
