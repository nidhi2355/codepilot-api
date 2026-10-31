import os
import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from agents.state import AgentState

logger = logging.getLogger(__name__)

class FileModification(BaseModel):
    file_path: str = Field(description="Relative path to the file being modified.")
    new_content: str = Field(description="The exact new content to write to the file.")

class CodingOutput(BaseModel):
    modifications: List[FileModification] = Field(description="List of files to modify with their new code.")
    explanation: str = Field(description="Brief explanation of the changes made.")

def coding_node(state: AgentState) -> dict:
    """
    LangGraph node: Generates code patches based on the plan and retrieved context,
    and writes them to the filesystem.
    """
    logger.info("Coding Agent: Generating code modifications...")
    
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )
    
    # Initialize the parser
    parser = JsonOutputParser(pydantic_object=CodingOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert autonomous software engineer. "
            "Follow the provided plan to fix the user's issue. "
            "Use the retrieved codebase context to ensure your changes are correct and import the right dependencies. "
            "Output the exact file paths and the FULL updated code for those files.\n\n"
            "{format_instructions}"
        )),
        ("human", "Task: {task}\n\nPlan: {plan}\n\nRetrieved Context: {context}")
    ])
    
    # Chain uses the LLM directly, then pipes output to the parser
    chain = prompt | llm | parser
    
    try:
        context_str = "\n\n".join([
            f"--- File: {chunk.get('file_path')} ---\n{chunk.get('content')}" 
            for chunk in state.get("retrieved_context", [])
        ])
        
        # We pass format_instructions from the parser into the prompt
        result_dict = chain.invoke({
            "task": state["task"],
            "plan": "\n".join(state.get("plan", [])),
            "context": context_str,
            "format_instructions": parser.get_format_instructions()
        })
        
        # The parser returns a dictionary, so we validate it back into our Pydantic model
        result = CodingOutput(**result_dict)
        
        modified_files = []
        for mod in result.modifications:
            os.makedirs(os.path.dirname(mod.file_path) or ".", exist_ok=True)
            with open(mod.file_path, "w", encoding="utf-8") as f:
                f.write(mod.new_content)
            modified_files.append(mod.file_path)
            logger.info(f"Successfully modified: {mod.file_path}")
            
        return {"modified_files": modified_files}
    
    except Exception as e:
        logger.error(f"Coding failed: {str(e)}")
        return {"errors": [f"Coding Error: {str(e)}"]}