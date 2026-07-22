import json
from typing import List, Dict, Optional, Literal, Any, TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# PERSONAS
# ==========================================
PersonaType = Literal["student", "parent", "policymaker", "general"]

# ==========================================
# PYDANTIC SCHEMAS
# ==========================================
class Entities(BaseModel):
    schools: List[str] = Field(default_factory=list)
    districts: List[str] = Field(default_factory=list)
    counties: List[str] = Field(default_factory=list)
    grade_levels: List[str] = Field(default_factory=list)
    subject_areas: List[str] = Field(default_factory=list)
    demographics: List[str] = Field(default_factory=list)

class Agent1Output(BaseModel):
    persona: PersonaType
    original_query: str
    clarification_needed: bool
    clarifying_question: Optional[str] = None
    intent_tags: List[str]
    entities: Entities
    sub_questions: List[str]

class RouteItem(BaseModel):
    sub_question: str
    data_source: Literal["internal_dataset", "web_search"]
    target_fields_or_query: str
    priority: int
    persona_weight_adjustment: Optional[Dict[str, float]] = None

class Agent2Plan(BaseModel):
    original_query: str
    persona: PersonaType
    retrieval_plan: List[RouteItem]

class InternalRecord(BaseModel):
    field_name: str
    value: Any
    data_year: str
    entity_name: str
    data_gap: bool = False
    gap_reason: Optional[str] = None

class WebRecord(BaseModel):
    query_used: str
    extracted_fact: str
    source_url: str
    publication_date: Optional[str] = None
    data_unavailable: bool = False

class Agent3MergedResult(BaseModel):
    internal_data: List[InternalRecord]
    web_data: List[WebRecord]

class PolicyRecommendation(BaseModel):
    title: str
    description: str
    policy_lever: str

class Agent4Synthesis(BaseModel):
    persona: PersonaType
    key_findings: List[str]
    ranked_options: Optional[List[Dict[str, Any]]] = None
    data_sources: List[str]
    data_gaps: List[str]
    policy_recommendations: List[PolicyRecommendation] = Field(default_factory=list)
    confidence_level: Literal["high", "medium", "low"]

class FinalChatResponse(BaseModel):
    persona: PersonaType
    formatted_response: str
    data_gap_notice: Optional[str] = None
    suggested_followup: Optional[str] = None

# ==========================================
# LANGGRAPH STATE
# ==========================================
class ChatbotState(TypedDict):
    persona: PersonaType
    raw_user_input: str
    agent1_output: Optional[dict]
    agent2_plan: Optional[dict]
    agent3_data: Optional[dict]
    agent4_synthesis: Optional[dict]
    final_response: Optional[dict]

# ==========================================
# LLM
# ==========================================
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

# ==========================================
# AGENT 1: Intake & Clarification
# ==========================================
agent1_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Agent 1 (Intake & Clarification) for an NC Education Equity Chatbot.
Your job is to parse the user's raw query and extract intent, entities, and sub-questions.

Persona context: {persona}

Output MUST be valid JSON adhering to this structure:
{{
  "persona": "{persona}",
  "original_query": "...",
  "clarification_needed": boolean,
  "clarifying_question": "string or null",
  "intent_tags": ["school_profile", "school_comparison", "county_overview", "metric_lookup", "policy_question", "general_education_equity", "news_or_recent_event"],
  "entities": {{
    "schools": [], "districts": [], "counties": [], "grade_levels": [], "subject_areas": [], "demographics": []
  }},
  "sub_questions": []
}}

RULES:
- If school/county is ambiguous, set clarification_needed = true and provide EXACTLY ONE clarifying question.
- Do NOT answer the query yet.
"""),
    ("user", "{raw_user_input}")
])

def run_agent_1(state: ChatbotState) -> ChatbotState:
    chain = agent1_prompt | llm.with_structured_output(Agent1Output)
    res = chain.invoke({"persona": state["persona"], "raw_user_input": state["raw_user_input"]})
    state["agent1_output"] = res.dict()
    return state

# ==========================================
# AGENT 2: Decomposition & Data Routing
# ==========================================
agent2_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Agent 2 (Decomposition & Routing Agent). Build an execution retrieval plan based on Agent 1's output.

Data Routing Rules:
1. Internal Dataset: Contains SPG scores (math/ELA/science), poverty rates, absentee rates, graduation rates, internet access, transportation scores, demographic breakdowns.
2. Web Search: Route here for recent news, state-level policy updates, NCES/DPI recent policy rules, general background topics.

Persona Weighting Strategy for Comparisons:
- Policymaker: Weights policy-relevant metrics (funding, internet access, state baseline gaps, poverty index).
- Parent: Weights school safety, graduation rates, internet access, student academic growth.
- Student: Weights access to resources, courses, internet, basic performance.
- General: Balanced weighting.

Output JSON conforming to Agent2Plan schema.
"""),
    ("user", "Agent 1 Payload: {agent1_payload}")
])

def run_agent_2(state: ChatbotState) -> ChatbotState:
    chain = agent2_prompt | llm.with_structured_output(Agent2Plan)
    res = chain.invoke({"agent1_payload": json.dumps(state["agent1_output"])})
    state["agent2_plan"] = res.dict()
    return state

# ==========================================
# AGENT 3A: Internal Data Retrieval
# ==========================================
def run_agent_3_internal(plan: dict) -> List[InternalRecord]:
    results = []
    for item in plan.get("retrieval_plan", []):
        if item["data_source"] == "internal_dataset":
            results.append(InternalRecord(
                field_name=item["target_fields_or_query"],
                value="84.2%",
                data_year="2023-2024",
                entity_name="Robeson County Schools",
                data_gap=False
            ))
    return results

# ==========================================
# AGENT 3B: Web Search Retrieval
# ==========================================
def run_agent_3_web(plan: dict) -> List[WebRecord]:
    results = []
    for item in plan.get("retrieval_plan", []):
        if item["data_source"] == "web_search":
            results.append(WebRecord(
                query_used=item["target_fields_or_query"],
                extracted_fact="NC DPI announced a $12M digital inclusion grant targeting tier 1 counties in late 2023.",
                source_url="https://www.dpi.nc.gov/news/broadband-grants-2023",
                publication_date="2023-11-15",
                data_unavailable=False
            ))
    return results

def run_agent_3(state: ChatbotState) -> ChatbotState:
    plan = state["agent2_plan"]
    internal_res = run_agent_3_internal(plan)
    web_res = run_agent_3_web(plan)
    merged = Agent3MergedResult(internal_data=internal_res, web_data=web_res)
    state["agent3_data"] = merged.dict()
    return state

# ==========================================
# AGENT 4: Synthesis & Policy Recommendation
# ==========================================
agent4_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Agent 4 (Synthesis & Policy Recommendation Agent).
Synthesize data from internal and web sources.

Persona: {persona}

Task Instructions:
1. Aggregate data and weight metrics according to persona priorities.
2. Flag data quality issues: outdated data years, missing data gaps, conflicts.
3. IF persona == 'policymaker': Generate 2-3 actionable policy recommendations using NC levers (e.g. NC Broadband Infrastructure Office, DPI Allotment Policy, PRC allocations).
4. IF persona == 'parent' or 'student': Translate metrics (SPG, absentee rates) into plain language context.
5. Set confidence_level (high/medium/low).

Return JSON matching Agent4Synthesis schema.
"""),
    ("user", "Retrieved Data: {agent3_payload}")
])

def run_agent_4(state: ChatbotState) -> ChatbotState:
    chain = agent4_prompt | llm.with_structured_output(Agent4Synthesis)
    res = chain.invoke({
        "persona": state["persona"],
        "agent3_payload": json.dumps(state["agent3_data"])
    })
    state["agent4_synthesis"] = res.dict()
    return state

# ==========================================
# AGENT 5: Response Formatting
# ==========================================
agent5_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Agent 5 (Response Formatting Agent).
Turn the Agent 4 synthesis into the final user-facing response.

Persona Tone Guidelines:
- Student: Casual, clear, motivating. Avoid jargon.
- Parent: Warm, practical, action-oriented. Focus on impact on their child.
- Policymaker: Precise, data-forward, professional. Lead with findings and policy recommendations.
- General: Balanced, informative, accessible.

Formatting Rules:
- Inline citations required (e.g., "per 2023 NC DPI data").
- Use short Markdown tables if comparing 3+ schools or metrics.
- Acknowledge data gaps honestly if present.
- Always include ONE optional, natural follow-up prompt at the end.

Return JSON matching FinalChatResponse schema.
"""),
    ("user", "Synthesis Payload: {agent4_payload}")
])

def run_agent_5(state: ChatbotState) -> ChatbotState:
    chain = agent5_prompt | llm.with_structured_output(FinalChatResponse)
    res = chain.invoke({"agent4_payload": json.dumps(state["agent4_synthesis"])})
    state["final_response"] = res.dict()
    return state