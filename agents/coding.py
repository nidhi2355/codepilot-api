import os
import logging
from typing import List
from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from agents.state import AgentState

logger = logging.getLogger(__name__)


class FileModification(BaseModel):
    file_path: str = Field(description="Relative path to the file being modified.")
    new_content: str = Field(description="The exact new content to write to the file.")


class CodingOutput(BaseModel):
    explanation: str = Field(
        default="",
        description="Brief explanation of the changes made.",
    )
    modifications: List[FileModification] = Field(
        description="List of files to modify with their new code."
    )


def coding_node(state: AgentState) -> dict:
    """
    LangGraph node: Generates code patches based on the plan and retrieved context,
    and writes them to the filesystem.
    """
    logger.info("Coding Agent: Generating code modifications...")

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )

    parser = JsonOutputParser(pydantic_object=CodingOutput)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are an expert autonomous software engineer. "
                "Follow the provided plan to fix the user's issue. "
                "Use the retrieved codebase context to ensure your changes are correct and import the right dependencies. "
                "Output the exact file paths and the FULL updated code for those files.\n\n"
                "{format_instructions}"
            ),
        ),
        (
            "human",
            "Task: {task}\n\nPlan: {plan}\n\nRetrieved Context: {context}\n\nPrevious Errors/Feedback: {errors}",
        ),
    ])

    chain = prompt | llm | parser

    context_str = "\n\n".join([
        f"--- File: {chunk.get('file_path')} ---\n{chunk.get('content')}"
        for chunk in state.get("retrieved_context", [])
    ])
    errors_list = list(state.get("errors", []))

    result = None
    last_error = None

    for attempt in range(2):
        try:
            result_dict = chain.invoke({
                "task": state["task"],
                "plan": "\n".join(state.get("plan", [])),
                "context": context_str,
                "errors": "\n".join(errors_list),
                "format_instructions": parser.get_format_instructions(),
            })
            result = CodingOutput(**result_dict)
            break
        except (ValidationError, ValueError, TypeError) as e:
            last_error = e
            logger.warning(f"CodingOutput validation failed on attempt {attempt + 1}: {e}")
            errors_list.append(
                f"Previous JSON output failed schema validation: {e}. "
                "Return valid JSON matching the exact schema with 'explanation' and 'modifications'."
            )
        except Exception as e:
            logger.error(f"Coding failed: {str(e)}")
            return {"errors": [f"Coding Error: {str(e)}"]}

    if result is None:
        logger.error(f"Coding failed after retries: {last_error}")
        return {"errors": [f"Coding Error: {str(last_error)}"]}

    try:
        repo_path = state.get("repo_path", ".")
        modified_files = []

        for mod in result.modifications:
            full_path = os.path.join(repo_path, mod.file_path)
            os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(mod.new_content)
            modified_files.append(mod.file_path)
            logger.info(f"Successfully modified: {full_path}")

        return {"modified_files": modified_files}

    except Exception as e:
        logger.error(f"Writing modified files failed: {str(e)}")
        return {"errors": [f"Coding Error: {str(e)}"]}