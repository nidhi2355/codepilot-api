import os
import logging
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from agents.state import AgentState

logger = logging.getLogger(__name__)

class ReflectionOutput(BaseModel):
    feedback: str = Field(description="Actionable feedback on why the test failed and how to fix the code.")

def reflection_node(state: AgentState) -> dict:
    """
    LangGraph node: Analyzes test failures and provides feedback for the next coding attempt.
    """
    logger.info("Reflection Agent: Analyzing test failures...")
    
    test_results = state.get("test_results", {})
    output = test_results.get("output", "")
    retry_count = state.get("retry_count", 0)
    
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )
    
    parser = JsonOutputParser(pydantic_object=ReflectionOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert senior software engineer reviewing a failing test pipeline. "
            "Analyze the test output and provide concise, actionable feedback on exactly what went wrong "
            "and how the coding agent should fix it in the next attempt.\n\n"
            "{format_instructions}"
        )),
        ("human", "Task: {task}\n\nTest Output:\n{test_output}")
    ])
    
    chain = prompt | llm | parser
    
    try:
        result_dict = chain.invoke({
            "task": state["task"],
            "test_output": output[-2000:],  # truncate just in case
            "format_instructions": parser.get_format_instructions()
        })
        
        feedback = result_dict.get("feedback", "No specific feedback generated.")
        logger.info(f"Reflection complete. Feedback: {feedback}")
        
        new_errors = state.get("errors", [])
        new_errors.append(f"Retry {retry_count + 1} Feedback: {feedback}")
        
        return {
            "retry_count": retry_count + 1,
            "errors": new_errors
        }
    except Exception as e:
        logger.error(f"Reflection failed: {str(e)}")
        return {"retry_count": retry_count + 1, "errors": [f"Reflection Error: {str(e)}"]}