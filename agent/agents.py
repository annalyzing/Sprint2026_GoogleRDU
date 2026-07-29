import os
import json
import re
import pandas as pd

from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv(
    dotenv_path=os.path.join(
        os.path.dirname(__file__),
        ".env"
    )
)


# ===============================
# GEMINI MODEL
# ===============================

api_key = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)


llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    temperature=0,
    max_tokens=1400,
    timeout=40,
    max_retries=1,
)



# ===============================
# DATASETS
# ===============================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATA_DIR = os.path.join(BASE_DIR, "data-files")

_cache = {}


DATASET_MAP = {

    "performance":
        "2025spg",

    "graduation":
        "2025gradrates",

    "attendance":
        "2025absent",

    "poverty":
        "frl-ratios",

    "teachers":
        "nc_county_attrition"
}



def load_dataset(name):

    print("LOADING DATASET:", name)

    if name in _cache:
        return _cache[name]

    csv_path = os.path.join(
        DATA_DIR,
        name + ".csv"
    )

    xlsx_path = os.path.join(
        DATA_DIR,
        name + ".xlsx"
    )

    if os.path.exists(csv_path):

        df = pd.read_csv(
            csv_path,
            low_memory=False
        )

    elif os.path.exists(xlsx_path):

        df = pd.read_excel(
            xlsx_path
        )

    else:

        print("DATASET NOT FOUND:", name)
        df = pd.DataFrame()


    _cache[name] = df

    return df



# ===============================
# EDUCATION KNOWLEDGE
# ===============================


def load_equity_context():

    path = os.path.join(
        DATA_DIR,
        "education_context.txt"
    )

    if os.path.exists(path):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            return f.read()


    return """
Education inequity refers to unequal access
to resources, opportunities, and outcomes.

Major contributors include:
- school funding differences
- broadband access
- transportation barriers
- poverty
- teacher shortages
- chronic absenteeism
- unequal access to advanced courses
"""



# ===============================
# DATA ROUTING
# ===============================


def detect_topics(question):

    q = question.lower()

    topics = []

    if any(x in q for x in [
        "score",
        "performance",
        "ranking",
        "rank",
        "grade",
        "top",
        "best",
        "highest",
        "lowest",
        "school"
    ]):
        topics.append("performance")


    if any(x in q for x in [
        "graduate",
        "graduation",
        "diploma"
    ]):
        topics.append("graduation")


    if any(x in q for x in [
        "attendance",
        "absent",
        "truancy"
    ]):
        topics.append("attendance")


    if any(x in q for x in [
        "poverty",
        "income",
        "free lunch",
        "frl"
    ]):
        topics.append("poverty")


    if any(x in q for x in [
        "teacher",
        "turnover",
        "shortage"
    ]):
        topics.append("teachers")


    return topics



def search_internal_datasets(question):

    topics = detect_topics(question)

    results = []

    limit = 10

    match = re.search(
        r"top\s+(\d+)",
        question.lower()
    )

    if match:
        limit = int(match.group(1))


    question_lower = question.lower()


    for topic in topics:

        filename = DATASET_MAP.get(topic)

        if not filename:
            continue


        df = load_dataset(filename)


        if df.empty:
            continue


        # clean columns
        df.columns = [
            str(c).strip()
            for c in df.columns
        ]


        filtered = df.copy()


        # -----------------------------
        # LOCATION FILTERING
        # -----------------------------

        location_words = []

        possible_locations = [
            "durham",
            "wake",
            "mecklenburg",
            "orange",
            "guilford",
            "forsyth",
            "buncombe",
            "johnston",
            "cumberland",
            "davidson",
            "alamance",
            "chatham"
        ]


        for word in possible_locations:

            if word in question_lower:
                location_words.append(word)



        # Apply location filter only if user specified one

        if location_words:


            for col in filtered.columns:


                if any(key in col.lower() for key in [
                    "county",
                    "district",
                    "lea",
                    "name",
                    "school"
                ]):


                    matches = filtered[
                        filtered[col]
                        .fillna("")
                        .astype(str)
                        .str.lower()
                        .apply(
                            lambda x:
                            any(
                                loc in x
                                for loc in location_words
                            )
                        )
                    ]


                    if not matches.empty:

                        filtered = matches

                        break



        # -----------------------------
        # RANKING LOGIC
        # -----------------------------


        score_col = None


        if "spg_score" in filtered.columns:

            score_col = "spg_score"


        else:

            score_columns = [
                c for c in filtered.columns
                if "score" in c.lower()
            ]


            if score_columns:
                score_col = score_columns[0]



        if score_col:


            filtered[score_col] = pd.to_numeric(
                filtered[score_col],
                errors="coerce"
            )


            filtered = filtered.sort_values(
                by=score_col,
                ascending=False
            )



        # -----------------------------
        # KEEP IMPORTANT COLUMNS
        # -----------------------------


        keep_cols = [

            c for c in filtered.columns

            if any(key in c.lower() for key in [

                "name",
                "school",
                "county",
                "district",
                "score",
                "grade",
                "rate",
                "percent",
                "pct",
                "span"

            ])

        ]


        if keep_cols:

            filtered = filtered[keep_cols]



        records = (

            filtered
            .head(limit)
            .fillna("")
            .to_dict(
                orient="records"
            )

        )


        results.append({

            "topic": topic,

            "records": records

        })



    return results


# ===============================
# PERSONA STYLE
# ===============================


PERSONAS = {


"student":
"""
Speak like a mentor.

Explain complicated ideas simply.
Avoid policy jargon.
Help the student understand why this matters.
Be encouraging.
""",


"parent":
"""
Speak like a school counselor.

Focus on how issues affect children.
Give practical explanations and possible actions.
""",


"policymaker":
"""
Speak like an education analyst.

Use evidence, trends, metrics,
and policy implications.
""",


"general":
"""
Explain clearly for someone learning
about education equity.
Balance examples and explanation.
""",


"map_parent":
"""
Explain NC map information to a parent.
Reference counties and schools when available.
Connect data to a child's experience.
""",


"map_student":
"""
Explain map information to a student.
Be direct, understandable, and encouraging.
""",


"map_policy":
"""
Analyze map information like a policy researcher.
Discuss patterns, disparities, and measurements.
"""

}



# ===============================
# RESPONSE FORMAT
# ===============================


class EducationResponse(BaseModel):

    formatted_response: str = Field(
        description="""
        Complete final answer.
        Must directly answer the question.
        Ranking questions require numbered lists or markdown tables.
        Include NC-specific examples whenever possible.
        """
    )

    suggested_followup: Optional[str] = Field(
        default=None,
        description="One useful follow-up question."
    )



# ===============================
# PROMPT
# ===============================


prompt = ChatPromptTemplate.from_messages([


(
"system",

"""
You are EduBridge AI.

You help users understand education inequity,
especially in North Carolina.

User persona:
{persona_style}

Education context:
{equity_context}

Is this a ranking question:
{is_ranking}

Your job:

- Explain education inequity clearly
- Analyze NC school data when available
- Connect problems to real causes
- Suggest possible solutions


IMPORTANT RULES:

1. Answer the user's exact question first.

2. If the user asks for:
- top schools
- best schools
- rankings
- highest/lowest performing schools

You MUST use this format:

Start with a short 1-2 sentence introduction.

Then provide a numbered list:

1. School Name — County/District
   - Why it stands out
   - Important programs, rankings, or opportunities

2. School Name — County/District
   - Why it stands out
   - Important programs, rankings, or opportunities

Continue until the requested number of schools is reached.

After the ranking list, include:

Equity Context

Explain:
- differences in school access
- AP/IB/CTE opportunities
- socioeconomic factors
- transportation or enrollment barriers

Do NOT use markdown tables for ranking questions.
Do NOT answer only as a paragraph.

3. If the user asks "why" or "how":

Use:

## Explanation

## NC Context

## What Can Be Done

4. Always include:
- NC school names when relevant
- NC county/district names when relevant
- programs/resources when available
- practical next steps

5. Use the dataset context as the source of truth.

6. If dataset information is unavailable, say:
"Based on available NC education information..."
and do not pretend a statistic came from the dataset.

7. Be specific. Avoid generic education explanations.

"""
),


(
"user",

"""
Conversation history:

{history}


User question:

{question}


Relevant NC dataset information:

{dataset}

"""
)

])
# ===============================
# MAIN FUNCTION
# ===============================


def process_education_query(
        message,
        persona="general",
        history=None):


    if history is None:
        history = []


    dataset = search_internal_datasets(message)

    print("DATASET FOUND:")
    print(json.dumps(dataset, indent=2)[:2000])


    history_text = "\n".join(
        [
            f"{x.get('role')}: {x.get('content')}"
            for x in history[-5:]
        ]
    )


    is_ranking = any(
        word in message.lower()
        for word in [
            "top",
            "best",
            "ranking",
            "highest",
            "lowest"
        ]
    )


    chain = (
        prompt
        |
        llm
    )


    response = chain.invoke({

        "question": message,

        "dataset": json.dumps(
            dataset,
            indent=2
        ),

        "history": history_text,

        "persona_style": PERSONAS.get(
            persona,
            PERSONAS["general"]
        ),

        "equity_context": load_equity_context(),

        "is_ranking": str(is_ranking)

    })


    answer = response.content


    if isinstance(answer, list):
        answer = "".join(
            item.get("text", "")
            for item in answer
            if isinstance(item, dict)
        )


    return str(answer)