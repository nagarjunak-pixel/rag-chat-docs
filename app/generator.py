"""
MODULE 5: LLM ANSWER GENERATOR
==============================
WHY THIS EXISTS:
  This is the "G" in RAG — Generation.
  We take the user's question + retrieved chunks and ask an LLM to answer
  FROM those chunks, not from its training data.
  
  This is what makes RAG different from just "chatting with ChatGPT":
  The LLM is constrained to answer from YOUR documents with citations.

WHAT IT DOES:
  1. Takes user question + retrieved context (from retriever.py)
  2. Builds a prompt: "Answer from this context. Cite sources. Say 'I don't know' if not here."
  3. Calls the LLM (GLM-5.2 via your cloud proxy)
  4. Returns the answer with source citations

WHY "answer ONLY from context"?
  Without this instruction, the LLM mixes its training data with your documents.
  It might say "Based on my knowledge..." and hallucinate facts that sound right
  but aren't in your files. This is the #1 RAG failure mode.
  The instruction "answer only from the provided context" anchors it to reality.

WHY cite sources?
  Trust. If the LLM says "Revenue was $2.3M (Source: report.pdf, page 5)",
  the user can verify by checking page 5. Without citations, the answer is
  just another opinion. With citations, it's verifiable.

WHY use GLM-5.2 cloud (via tryai.myqrmart.com)?
  - It's your current model, already configured, no API key needed
  - No rate limits (we tested this earlier)
  - OpenAI-compatible API format — standard interface
  - Can be swapped for Ollama local model, OpenAI, or Anthropic by changing one URL
"""

import requests
import json
from typing import List, Dict


# WHY these constants? They connect to your cloud LLM proxy.
# This is the same model Hermes uses (GLM-5.2:cloud via tryai.myqrmart.com).
# OpenAI-compatible API format — standard chat completions endpoint.
LLM_URL = "https://tryai.myqrmart.com/v1/chat/completions"
LLM_MODEL = "glm-5.2:cloud"


# WHY a system prompt template?
# The system prompt is the MOST IMPORTANT part of RAG.
# It tells the LLM:
#   1. Your job (answer from context)
#   2. Your constraints (cite sources, don't hallucinate)
#   3. What to do when you don't know (say "I don't know")
# Without a good system prompt, RAG fails. The LLM will mix in training data,
# make things up, and give confident wrong answers.
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context documents.

RULES:
1. Answer the question using ONLY the information in the provided context.
2. If the answer is not in the context, say "I don't know based on the provided documents."
3. Always cite your sources. After each fact, write (Source: filename, page X).
4. Do NOT use your training data or general knowledge — only the context below.
5. If multiple sources support a fact, cite all of them.
6. Be concise but complete. Don't pad the answer with irrelevant information.

FORMAT:
Write a clear answer. After each claim, cite the source in parentheses.
At the end, list all sources you referenced.

Example answer:
"The company's revenue model is based on subscription fees (Source: report.pdf, page 5). 
Q4 revenue was $2.3M, up 20% year-over-year (Source: notes.txt, page 0).

Sources cited:
- report.pdf, page 5
- notes.txt, page 0"
"""


class AnswerGenerator:
    """Generates answers from retrieved context using an LLM."""
    
    def __init__(self, model: str = LLM_MODEL, url: str = LLM_URL):
        self.model = model
        self.url = url
        print(f"  🧠 Answer generator ready — model: {model}")
    
    def generate(self, question: str, context: str) -> str:
        """
        Generate an answer from the question + retrieved context.
        
        THE FLOW:
          1. Build messages array: [system_prompt, context_as_user_message]
          2. Call LLM API (OpenAI-compatible chat completions)
          3. Return the LLM's answer
          
        WHY this message structure?
          - System message: sets the rules (answer from context, cite sources)
          - User message: combines the question + retrieved context
          
          The LLM sees: "Here's the context [retrieved chunks]. Here's the question. 
          Answer from context only."
          
        Args:
            question: User's question
            context: Formatted context from retriever.retrieve_with_context()
            
        Returns:
            LLM's answer with citations
        """
        # WHY combine question + context in one user message?
        # The LLM needs to see both at the same time to answer from context.
        # If we sent them as separate messages, the model might lose track
        # of which is the question vs the reference material.
        user_message = f"""CONTEXT DOCUMENTS:
{context}

QUESTION: {question}

Answer the question using ONLY the context above. Cite sources."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        # WHY this API call format? It's the standard OpenAI chat completions format.
        # Any OpenAI-compatible API (OpenAI, Anthropic via proxy, Ollama, GLM, etc.)
        # uses this exact format. That's why we can swap models by changing one URL.
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.3,  # WHY 0.3? Low temperature = more factual, less creative
                "max_tokens": 1000
            },
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        
        return answer
    
    def generate_stream(self, question: str, context: str):
        """
        Stream the answer token by token.
        
        WHY streaming? For the web app, we want the user to see the answer
        appear word-by-word as it's generated. This makes the app feel fast
        and responsive. Nobody wants to stare at a loading spinner for 10 seconds.
        
        This is a generator function — it yields chunks as they arrive.
        """
        user_message = f"""CONTEXT DOCUMENTS:
{context}

QUESTION: {question}

Answer the question using ONLY the context above. Cite sources."""
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        # WHY stream=True? Tells the API to return chunks as they're generated
        # instead of waiting for the full response. We yield each chunk.
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 1000,
                "stream": True
            },
            headers={"Content-Type": "application/json"},
            timeout=60,
            stream=True
        )
        
        # WHY parse SSE (Server-Sent Events)? The streaming API returns
        # data as "data: {json}\n\n" lines. We parse each line, extract the
        # text delta, and yield it.
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except:
                        continue


# WHY this test block? Students can verify the LLM answers from context
# and produces citations before wiring up the full pipeline.
if __name__ == "__main__":
    print("\n=== Answer Generator Test ===\n")
    
    generator = AnswerGenerator()
    
    # Simulate retrieved context
    test_context = """[Source 1: report.pdf, page 5, relevance: 0.89]
The company's revenue model is based on subscription fees charged monthly. 
Q4 2025 revenue was $2.3M, up 20% year-over-year.

[Source 2: notes.txt, page 0, relevance: 0.75]
The board approved a 15% increase in marketing spend for Q1 2026."""
    
    question = "What is the revenue model and how much revenue did they make?"
    print(f"  ❓ Question: {question}")
    print(f"  📄 Context: (simulated)")
    
    answer = generator.generate(question, test_context)
    print(f"\n  💬 Answer:\n{answer}")