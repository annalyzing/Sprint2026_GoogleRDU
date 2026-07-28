from dotenv import load_dotenv
load_dotenv()
from agents import run_agent_1, run_agent_2, run_agent_3, run_agent_4, run_agent_5
def run_agent(message: str, persona: str, history: list = [])-> str:
    state = {
        "persona": persona,
        "raw_user_input": message,
        "agent1_output": None,
        "agent2_plan": None,
        "agent3_data": None,
        "agent4_synthesis": None,
        "final_response": None,
    }
    state = run_agent_1(state)
    if state["agent1_output"].get("clarification_needed"):
        return state["agent1_output"]["clarifying_question"]
    state = run_agent_2(state)
    state = run_agent_3(state)
    state = run_agent_4(state)
    state = run_agent_5(state)
    final = state["final_response"]
    response = final["formatted_response"]
    if final.get("data_gap_notice"):
        response += f"\n\n*Note: {final['data_gap_notice']}*"
    if final.get("suggested_followup"):
        response += f"\n\n💡 *{final['suggested_followup']}*"
    return response
