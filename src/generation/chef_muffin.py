"""
chef_muffin.py

RAG generation step for the "Chef Muffin" assistant.

This module:
1) Retrieves relevant muffin recipes from the vector store.
2) Builds a context block from retrieved recipes.
3) Generates a French response using an LLM with strict muffin-only guardrails.

The LLM backend is intentionally abstracted behind a simple function so it can
be switched later (OpenAI/Azure, Mistral local, etc.).
"""

from __future__ import annotations
import re 

from dataclasses import dataclass
from typing import List

from src.retrieval.retrieve import retrieve
from src.generation.ollama_client import generate as ollama_generate

SYSTEM_PROMPT = """Tu es CHEF MUFFIN, un assistant culinaire obsessionnel mais sympathique.

Directives:
1) Tu ne proposes QUE des muffins (sucrés ou salés). Si la demande n'est pas un muffin, refuse poliment et ramène le sujet au muffin.
2) Utilise UNIQUEMENT les recettes fournies dans [CONTEXTE]. N'invente pas d'ingrédients ou de recettes.
3) Choisis toujours la recette la plus proche de la demande, même si tous les ingrédients ne correspondent pas exactement.
4) Réponds toujours de manière à donner les ingrédients nécessaires et après les étapes de la recette.
5) N'écris jamais de phrases d'excuse comme "je suis désolé" ou "je ne peux pas".
6) Réponds toujours en français, de façon claire, confiante et appétissante.
"""


MUFFIN_ALLOWED = re.compile(r"\b(muffin|muffins|cupcake|cupcakes)\b", re.IGNORECASE)

@dataclass
class RecipeHit:
    title: str
    ingredients: str


def build_context(hits: List[dict]) -> str:
    items = []
    for h in hits:
        title = (h.get("title") or "").strip()
        ingredients = (h.get("ingredients") or "").strip()
        if not title and not ingredients:
            continue
        items.append(f"- Titre: {title}\n  Ingrédients: {ingredients}")
    return "\n".join(items)


def build_prompt(user_query: str, context: str) -> str:
    return f"""{SYSTEM_PROMPT}

[CONTEXTE]
{context}

[QUESTION]
{user_query}
"""


def llm_generate_placeholder(prompt: str) -> str:
    """
    Placeholder LLM call.

    For now, this returns a deterministic stub so the pipeline runs
    without external dependencies.
    """
    return (
        "Je peux t'aider uniquement avec des muffins. "
        "D'après le contexte fourni, voici une suggestion de muffin adaptée à ta demande:\n\n"
        "(Branche le modèle de génération pour obtenir une réponse complète.)"
    )

def llm_generate(prompt: str) -> str:
    return ollama_generate(prompt, model="mistral")

def answer(user_query: str, top_k: int = 3) -> str:
    if not MUFFIN_ALLOWED.search(user_query):
        return (
            "Je peux uniquement aider sur des muffins (sucrés ou salés). "
            "Dis-moi quels ingrédients tu as, et je te proposerai un muffin adapté."
        )

    hits = retrieve(user_query, top_k=top_k)
    context = build_context(hits)
    prompt = build_prompt(user_query, context)
    return llm_generate(prompt)



if __name__ == "__main__":
    q = "Je veux un muffin au chocolat avec des oeufs"
    print(answer(q, top_k=3))
