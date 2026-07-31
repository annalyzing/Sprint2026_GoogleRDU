import os
import json
import pandas as pd
from typing import List, Dict, Optional, Any
import re

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ===============================
# GEMINI MODEL SETUP
# ===============================

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    temperature=0.2,
    max_tokens=1000,
    timeout=40,
    max_retries=2,
)

# ===============================
# DATASETS & CACHING
# ===============================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data-files")

_cache: Dict[str, pd.DataFrame] = {}

DATASET_MAP = {
    "performance": "2025spg",
    "graduation": "2025gradrates",
    "attendance": "2025absent",
    "poverty": "frl-ratios",
    "teachers": "nc_county_attrition",
}


def load_dataset(name: str) -> pd.DataFrame:
    """Load a dataset into memory and reuse it after the first load."""
    if name in _cache:
        return _cache[name]

    csv_path = os.path.join(DATA_DIR, f"{name}.csv")
    xlsx_path = os.path.join(DATA_DIR, f"{name}.xlsx")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, low_memory=False)
    elif os.path.exists(xlsx_path):
        df = pd.read_excel(xlsx_path)
    else:
        df = pd.DataFrame()

    _cache[name] = df
    return df


def load_equity_context() -> str:
    """Load supplemental education knowledge context."""
    path = os.path.join(DATA_DIR, "education_context.txt")

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as file:
            return file.read()

    return """
Education inequity refers to unequal access to resources, opportunities, and outcomes.
Major contributors in North Carolina include school funding differences, broadband access,
transportation barriers, poverty, teacher shortages, chronic absenteeism, and unequal
access to advanced courses.
"""


# ===============================
# USER PROFILE RULES
# ===============================
PERSONA_INSTRUCTIONS = {
    "student": """
AUDIENCE: Student.

Use friendly, simple language at about an eighth-grade reading level.
Explain unfamiliar terms in plain language and focus on what the answer means for a student.

RESPONSE STYLE:
- Give a direct answer first.
- Write 1 to 3 short paragraphs, with no paragraph longer than 3 sentences.
- Use bullets only if the question asks for a list, steps, comparisons, or examples.
- Maximum: 110 words.
""",

    "policymaker": """
AUDIENCE: Policymaker.

Use a formal, concise, decision-focused tone.
Prioritize relevant North Carolina data, comparisons, trends, equity implications, and
practical policy options. Assume familiarity with standard education terminology.

RESPONSE STYLE:
- Give the key finding first.
- Write 1 to 3 short paragraphs, with no paragraph longer than 3 sentences.
- Use bullets only for rankings, multiple recommendations, or several statistics.
- Maximum: 170 words.
""",

    "educator": """
AUDIENCE: Educator.

Use a practical, collaborative tone.
Connect the answer to student learning, school supports, attendance, and classroom or
school-level action.

RESPONSE STYLE:
- Give the key takeaway first.
- Write 1 to 3 short paragraphs, with no paragraph longer than 3 sentences.
- Use bullets only when they make actions or comparisons easier to scan.
- Maximum: 140 words.
""",

    "parent": """
AUDIENCE: Parent or guardian.

Use warm, plain, supportive language.
Explain what the answer may mean for a child or family without using unnecessary jargon.

RESPONSE STYLE:
- Give a clear answer first.
- Write 1 to 3 short paragraphs, with no paragraph longer than 3 sentences.
- Use bullets only if they make next steps clearer.
- Maximum: 125 words.
""",

    "general": """
AUDIENCE: General public.

Use clear, balanced, practical language.

RESPONSE STYLE:
- Answer directly before providing context.
- Write 1 to 3 short paragraphs, with no paragraph longer than 3 sentences.
- Use bullets only when a list is actually the clearest format.
- Maximum: 130 words.
""",
}
# ===============================
# DYNAMIC INTENT ANALYSIS
# ===============================

class QueryIntent(BaseModel):
    topics: List[str] = Field(
        description=(
            "Relevant dataset topics from: performance, graduation, attendance, "
            "poverty, teachers"
        )
    )
    counties_or_districts: List[str] = Field(
        default=[],
        description="Explicit North Carolina county or district names in the question",
    )
    is_ranking: bool = Field(
        default=False,
        description="True when the user asks for top, bottom, best, worst, or rankings",
    )
    sort_ascending: bool = Field(
        default=False,
        description=(
            "True for lowest/worst rankings. False for top/best/highest-score rankings."
        ),
    )
    limit: int = Field(
        default=10,
        description="Number of requested records; use 10 if unspecified",
    )


def analyze_intent(question: str) -> QueryIntent:
    """Use Gemini to identify the datasets and filters relevant to a question."""
    try:
        structured_llm = llm.with_structured_output(QueryIntent)

        return structured_llm.invoke(
            "Analyze this North Carolina education question and determine data lookup "
            f"parameters:\n\n{question}"
        )
    except Exception as error:
        print(f"[Intent Analysis Fallback]: {error}")

        return QueryIntent(
            topics=[],
            counties_or_districts=[],
            is_ranking=False,
            sort_ascending=False,
            limit=10,
        )


# ===============================
# DYNAMIC DATA RETRIEVAL ENGINE
# ===============================

def search_internal_datasets(question: str) -> List[Dict[str, Any]]:
    """Filter and sort relevant datasets based on the detected question intent."""
    intent = analyze_intent(question)
    results = []

    for topic in intent.topics:
        filename = DATASET_MAP.get(topic)

        if not filename:
            continue

        df = load_dataset(filename)

        if df.empty:
            continue

        filtered = df.copy()
        filtered.columns = [str(column).strip() for column in filtered.columns]

        # Filter by county, district, school, or LEA when one is mentioned.
        if intent.counties_or_districts:
            location_columns = [
                column
                for column in filtered.columns
                if any(
                    keyword in column.lower()
                    for keyword in ["county", "district", "lea", "name", "school"]
                )
            ]

            if location_columns:
                mask = pd.Series(False, index=filtered.index)

                for column in location_columns:
                    column_text = filtered[column].fillna("").astype(str).str.lower()

                    for location in intent.counties_or_districts:
                        mask |= column_text.str.contains(
                            location.lower(),
                            regex=False,
                        )

                matched_df = filtered[mask]

                if not matched_df.empty:
                    filtered = matched_df

        # Find a likely numeric metric for sorting.
        sort_column = None

        for candidate in [
            "spg_score",
            "score",
            "rate",
            "percent",
            "pct",
            "attrition",
        ]:
            matching_columns = [
                column
                for column in filtered.columns
                if candidate in column.lower()
            ]

            if matching_columns:
                sort_column = matching_columns[0]
                break

        if sort_column:
            filtered[sort_column] = pd.to_numeric(
                filtered[sort_column],
                errors="coerce",
            )
            filtered = filtered.sort_values(
                by=sort_column,
                ascending=intent.sort_ascending,
            )

        # Keep useful columns so the model receives concise data.
        keep_columns = [
            column
            for column in filtered.columns
            if any(
                keyword in column.lower()
                for keyword in [
                    "name",
                    "school",
                    "county",
                    "district",
                    "score",
                    "grade",
                    "rate",
                    "percent",
                    "pct",
                    "span",
                    "attrition",
                ]
            )
        ]

        if keep_columns:
            filtered = filtered[keep_columns]

        records = (
            filtered.head(intent.limit)
            .fillna("")
            .to_dict(orient="records")
        )

        results.append({
            "topic": topic,
            "records": records,
        })

    return results


# ===============================
# RESPONSE CLEANING
# ===============================

def clean_model_output(response_obj: Any) -> str:
    """Extract the user-facing answer and remove accidental model meta-text."""
    if isinstance(response_obj, str):
        text = response_obj
    else:
        content = response_obj.content if hasattr(response_obj, "content") else response_obj

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = []

            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
                elif isinstance(block, str):
                    parts.append(block)

            text = "\n".join(parts)
        else:
            text = str(content)

    # Removes common accidental Gemini "meta" lines before the real answer.
    cleaned_lines = []

    for line in text.splitlines():
        label = line.strip().strip("*").strip().lower().rstrip(":")

        is_meta_line = (
            label.startswith("/repeats")
            or label.startswith("repeats")
            or label in {
                "checked",
                "checked.",
                "final polish",
                "final answer",
                "response complete",
                "final response",
            }
        )

        if not is_meta_line:
            cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned

# ===============================
# PROMPT ENGINE
# ===============================

SYSTEM_PROMPT = """
You are EduBridge AI, a helpful expert on North Carolina education equity, policy,
and student outcomes.

Answer the user's actual question naturally and directly. Follow the selected
Audience Profile Instructions for vocabulary, tone, data depth, and length.

IMPORTANT:
- Return only the answer the user should see.
- Never mention prompts, formatting rules, profiles, checking, polishing, or compliance.
- Never write phrases such as "Checked," "Final Polish," "Final Answer," or "/repeats".
- For simple questions such as "Why is attendance important?", give a natural explanation
  in one or two short paragraphs. Do not turn it into a report.
- Use bullets only when the question genuinely asks for a list, ranking, comparison,
  steps, or several recommendations.
- Write at most 3 short paragraphs. Each paragraph must be no more than 3 sentences.
- Do not invent statistics. Use the provided dataset only when it adds useful evidence.
"""
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", """
Audience Profile Instructions:
{persona_instructions}

Background Education Knowledge:
{equity_context}

Relevant North Carolina Dataset Retrieval:
{dataset}

Conversation History:
{history}

User Question:
{question}
"""),
])


# ===============================
# MAIN PROCESSING FUNCTION
# ===============================

def process_education_query(
    message: str,
    persona: str = "general",
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Process a question and tailor the answer to the selected user profile."""
    if history is None:
        history = []

    # Makes "Policy Maker", "policy-maker", and "policymaker" work similarly.
    persona_key = persona.strip().lower().replace(" ", "").replace("-", "")

    persona_aliases = {
        "policymaker": "policymaker",
        "student": "student",
        "educator": "educator",
        "teacher": "educator",
        "parent": "parent",
        "guardian": "parent",
        "general": "general",
    }

    selected_persona = persona_aliases.get(persona_key, "general")
    persona_instructions = PERSONA_INSTRUCTIONS[selected_persona]

    dataset_records = search_internal_datasets(message)
    dataset_json = (
        json.dumps(dataset_records, indent=2)
        if dataset_records
        else "No direct internal dataset matches."
    )

    history_text = (
        "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}"
            for item in history[-5:]
        )
        if history
        else "No previous chat history."
    )

    chain = prompt_template | llm

    raw_response = chain.invoke({
        "question": message,
        "dataset": dataset_json,
        "history": history_text,
        "equity_context": load_equity_context(),
        "persona_instructions": persona_instructions,
    })

    cleaned_response = clean_model_output(raw_response)

    return limit_to_four_paragraphs(cleaned_response)