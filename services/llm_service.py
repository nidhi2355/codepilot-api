import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class LLMService:
    @staticmethod
    def generate_answer(query: str, retrieved_chunks: list) -> str:
        # 1. Format the retrieved code chunks into a readable string for the prompt
        context_text = ""
        for chunk in retrieved_chunks:
            context_text += f"\n--- File: {chunk['file_path']} ---\n{chunk['content']}\n"

        # 2. Build the system prompt
        system_prompt = (
            "You are CodePilot, an expert AI software engineer. "
            "Answer the user's question based strictly on the provided codebase context. "
            "Provide brief code snippets if they help explain the solution. "
            "If the context does not contain the answer, state that clearly without guessing."
        )

        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

        # 3. Execute inference
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1024
        )

        return completion.choices[0].message.content