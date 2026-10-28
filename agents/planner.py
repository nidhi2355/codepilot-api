import os
import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from agents.state import AgentState

logger = logging.getLogger(__name__)

class PlanOutput(BaseModel):
    steps: List[str] = Field(description="A step-by-step actionable plan to solve the engineering task.")

def plan_node(state: AgentState) -> dict:
    """
    LangGraph node: Converts a user request into a step-by-step engineering plan.
    """
    logger.info("Planner Agent: Generating execution plan...")
    
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert software architect. Break down the following software "
            "engineering task into a clear, concise step-by-step plan. "
            "Keep steps actionable (e.g., '1. Inspect auth.py', '2. Add JWT utilities'). "
            "Do not output actual code, only the structured plan steps."
        )),
        ("human", "Task: {task}")
    ])
    
    structured_llm = llm.with_structured_output(PlanOutput)
    chain = prompt | structured_llm
    
    try:
        result = chain.invoke({"task": state["task"]})
        logger.info(f"Generated plan with {len(result.steps)} steps.")
        return {"plan": result.steps}
    except Exception as e:
        logger.error(f"Planning failed: {str(e)}")
        return {"plan": [], "errors": [f"Planner Error: {str(e)}"]}