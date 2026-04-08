"""Reasoning prompt templates for Deep Memory.

Inspired by Honcho's formal logic framework:
- Explicit: extract stated facts
- Deductive: derive certain conclusions from premises
- Inductive: identify patterns across multiple observations
- Abductive: infer simplest explanations for behavior
"""

ENTITY_EXTRACTION_PROMPT = """\
Analyze the following conversation and identify all entities (people, projects, \
concepts, organizations) that are meaningfully discussed — not just mentioned in passing.

Return a JSON array of entities:
```json
[
  {{"name": "Entity Name", "type": "person|project|concept|organization", "relevance": "why this entity matters in the conversation"}}
]
```

If no meaningful entities are found, return an empty array: []

## Conversation
{transcript}
"""

SESSION_REASONING_PROMPT = """\
You are a reasoning engine that extracts structured insights from conversations.
Analyze the following conversation focusing on **{entity_name}**.

Extract insights in these categories:

1. **Explicit** — Facts directly stated about {entity_name}
2. **Deductive** — Certain conclusions derivable from stated premises
3. **Inductive** — Patterns observed across this and prior knowledge
4. **Abductive** — Simplest explanations for {entity_name}'s behavior/preferences

## Rules
- Only include insights with genuine informational value
- Deductive conclusions MUST have clear premises
- Inductive patterns need at least 2 supporting observations
- Abductive inferences should be the *simplest* explanation, not speculation
- Skip trivial or obvious observations
- If existing conclusions contradict new evidence, flag them

## Existing knowledge about {entity_name}
{existing_conclusions}

## Conversation
{transcript}

Return a JSON object:
```json
{{
  "explicit": [
    {{"content": "stated fact", "confidence": 0.0-1.0}}
  ],
  "deductive": [
    {{"premises": ["premise 1", "premise 2"], "conclusion": "derived conclusion", "confidence": 0.0-1.0}}
  ],
  "inductive": [
    {{"observations": ["obs 1", "obs 2"], "pattern": "identified pattern", "confidence": 0.0-1.0}}
  ],
  "abductive": [
    {{"behavior": "observed behavior", "explanation": "simplest explanation", "confidence": 0.0-1.0}}
  ],
  "contradictions": [
    {{"existing": "old conclusion", "new_evidence": "what contradicts it", "resolution": "which is more likely correct"}}
  ],
  "card_updates": {{
    "name": "...",
    "role": "...",
    "preferences": ["..."],
    "key_facts": ["..."]
  }}
}}
```

Omit empty categories. Only include card_updates if there's genuinely new biographical info.
"""

SESSION_SUMMARY_PROMPT = """\
Summarize the following conversation concisely.

Provide:
1. **short_summary** — 1-2 sentences capturing the essence
2. **key_decisions** — List of decisions made or conclusions reached (if any)
3. **entities_mentioned** — Names of people, projects, or concepts discussed

## Conversation
{transcript}

Return JSON:
```json
{{
  "short_summary": "...",
  "key_decisions": ["decision 1", "decision 2"],
  "entities_mentioned": ["entity 1", "entity 2"]
}}
```
"""

CONSOLIDATION_PROMPT = """\
You are reviewing existing conclusions about **{entity_name}** to identify \
redundancies, contradictions, and opportunities for consolidation.

## Current conclusions
{conclusions}

Analyze and return:
```json
{{
  "redundant_pairs": [
    {{"ids": [id1, id2], "keep": id1, "reason": "why this one is better"}}
  ],
  "contradictions": [
    {{"ids": [id1, id2], "resolution": "which is correct and why", "keep": id1}}
  ],
  "consolidated": [
    {{"from_ids": [id1, id2, id3], "new_content": "merged/upgraded conclusion", "type": "inductive", "confidence": 0.9}}
  ]
}}
```

Omit empty categories. Only flag genuine issues, not minor wording differences.
"""
