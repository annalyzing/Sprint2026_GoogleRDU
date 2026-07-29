import os
import json
import re
import pandas as pd
from typing import List, Dict, Optional, Any
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
    temperature=0.4,
    max_tokens=2000,
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
    "teachers": "nc_county_attrition"
}

def load_dataset(name: str) -> pd.DataFrame:
    """Load dataset into memory with caching."""
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
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    return """
Education inequity refers to unequal access to resources, opportunities, and outcomes.
Major contributors in NC include: school funding differences, broadband access,
transportation barriers, poverty, teacher shortages, chronic absenteeism, and unequal access to advanced courses.
"""

# ===============================
# DYNAMIC INTENT ANALYSIS (STRUCTURED OUTPUT)
# ===============================

class QueryIntent(BaseModel):
    topics: List[str] = Field(
        description="Relevant dataset topics from: 'performance', 'graduation', 'attendance', 'poverty', 'teachers'"
    )
    counties_or_districts: List[str] = Field(
        default=[],
        description="List of explicit NC county or district names mentioned (e.g., ['Durham', 'Wake'])"
    )
    is_ranking: bool = Field(
        default=False,
        description="True if the user is asking for top/bottom/best/worst rankings or lists"
    )
    sort_ascending: bool = Field(
        default=False,
        description="True if asking for 'lowest', 'worst', 'highest absenteeism', etc. False if 'top', 'best', 'highest scores'."
    )
    limit: int = Field(
        default=10,
        description="Number of records requested (default 10 if unspecified)"
    )

def analyze_intent(question: str) -> QueryIntent:
    """Uses Gemini to semantically understand intent instead of brittle string regex."""
    try:
        structured_llm = llm.with_structured_output(QueryIntent)
        return structured_llm.invoke(
            f"Analyze this education question and determine data lookup parameters:\n\"{question}\""
        )
    except Exception as e:
        print(f"[Intent Analysis Fallback]: {e}")
        return QueryIntent(topics=[], counties_or_districts=[], is_ranking=False, sort_ascending=False, limit=10)

# ===============================
# DYNAMIC DATA RETRIEVAL ENGINE
# ===============================

def search_internal_datasets(question: str) -> List[Dict[str, Any]]:
    """Dynamically filters and sorts datasets based on LLM-extracted intent."""
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
        filtered.columns = [str(c).strip() for c in filtered.columns]

        # 1. Dynamic Location Filtering across relevant columns
        if intent.counties_or_districts:
            location_cols = [
                c for c in filtered.columns 
                if any(k in c.lower() for k in ["county", "district", "lea", "name", "school"])
            ]
            
            if location_cols:
                mask = pd.Series(False, index=filtered.index)
                for col in location_cols:
                    col_str = filtered[col].fillna("").astype(str).str.lower()
                    for loc in intent.counties_or_districts:
                        mask |= col_str.str.contains(loc.lower(), regex=False)
                
                matched_df = filtered[mask]
                if not matched_df.empty:
                    filtered = matched_df

        # 2. Dynamic Sorting based on topic metrics
        sort_col = None
        for candidate in ["spg_score", "score", "rate", "percent", "pct", "attrition"]:
            matching_cols = [c for c in filtered.columns if candidate in c.lower()]
            if matching_cols:
                sort_col = matching_cols[0]
                break

        if sort_col:
            filtered[sort_col] = pd.to_numeric(filtered[sort_col], errors="coerce")
            filtered = filtered.sort_values(
                by=sort_col, 
                ascending=intent.sort_ascending
            )

        # 3. Intelligent Column Retention
        keep_cols = [
            c for c in filtered.columns
            if any(k in c.lower() for k in [
                "name", "school", "county", "district", "score", 
                "grade", "rate", "percent", "pct", "span", "attrition"
            ])
        ]
        if keep_cols:
            filtered = filtered[keep_cols]

        records = filtered.head(intent.limit).fillna("").to_dict(orient="records")
        results.append({
            "topic": topic,
            "records": records
        })

    return results

# ===============================
# RESPONSE SANITIZER
# ===============================

def clean_model_output(response_obj: Any) -> str:
    """
    Extracts purely the clean text payload from Gemini/LangChain responses,
    filtering out internal API metadata, signatures, and raw dictionary wrappers.
    """
    if isinstance(response_obj, str):
        return response_obj.strip()

    if hasattr(response_obj, "content"):
        content = response_obj.content
    else:
        content = response_obj

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    text_parts.append(str(block["text"]))
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts).strip()

    return str(content)

# ===============================
# PROMPT ENGINE
# ===============================

SYSTEM_PROMPT = """You are EduBridge AI, an adaptable, highly knowledgeable, and empathetic expert on education equity, policy, and student outcomes in North Carolina.

YOUR GOAL:
Act like a natural, fluid conversational assistant (like ChatGPT) dedicated to education equity.

FORMATTING RULES:
1. Write in clear, standard paragraphs and natural bullet points.
2. DO NOT use horizontal dividers or lines (never use '---').
3. DO NOT add unnecessary empty lines or extra line breaks between paragraphs.
4. Use standard headings (like ###) only when introducing major new sections.
5. Keep bullet points tight and directly under their section without extra line gaps.

CONTENT GUIDELINES:
1. Direct & Fluid: Answer the user's core question immediately.
2. Grounding: Use the provided NC Dataset context as your accurate data source. If specific statistics aren't available, rely on general education equity principles transparently.
3. Actionable Depth: Connect data points back to systemic causes and actionable solutions.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", """
Background Education Knowledge:
{equity_context}

Relevant NC Dataset Retrieval:
{dataset}

Conversation History:
{history}

User Question:
{question}
""")
])

# ===============================
# MAIN PROCESSING FUNCTION
# ===============================

def process_education_query(
    message: str,
    persona: str = "general",
    history: Optional[List[Dict[str, str]]] = None
) -> str:
    """Processes incoming user queries with dynamic intent extraction, flexible response generation, and output cleaning."""
    if history is None:
        history = []

    # 1. Search datasets using dynamic intent
    dataset_records = search_internal_datasets(message)
    dataset_json = json.dumps(dataset_records, indent=2) if dataset_records else "No direct internal dataset matches."

    # 2. Format history
    history_text = "\n".join([
        f"{x.get('role', 'user')}: {x.get('content', '')}" 
        for x in history[-5:]
    ]) if history else "No previous chat history."

    # 3. Execute LLM chain
    chain = prompt_template | llm

    raw_response = chain.invoke({
        "question": message,
        "dataset": dataset_json,
        "history": history_text,
        "equity_context": load_equity_context()
    })

    # 4. Clean raw payload (strips signatures, dictionaries, and extras metadata)
    return clean_model_output(raw_response)
