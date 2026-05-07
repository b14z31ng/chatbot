---
name: rag-project-plan
description: 'Create industry-standard PROJECT_PLAN.md for AI chatbot with RAG. Use when user needs complete plan with architecture, phases, pipeline, API, testing, deployment, standards.'
argument-hint: 'Provide domain, constraints, stack prefs, auth model, deployment target, knowledge sources.'
user-invocable: true
---

# RAG Project Plan Generator

## When to Use
- User requests a complete PROJECT_PLAN.md for a RAG chatbot system
- Need a structured, implementation-ready plan with clear sections and checklists

## Inputs
- Domain and use case
- Constraints: hosting, auth, compliance, latency, cost
- Preferred stack or cloud
- Knowledge sources and formats

## Output
- PROJECT_PLAN.md in workspace root
- Plan is concise, production-grade, and grounded

## Procedure (Quick Checklist)
1. Confirm scope: RAG only, no fine-tuning; answer only from KB; unknown handling required.
2. Define tech stack with justification.
3. Describe system architecture with text diagram and data flow.
4. Provide production-grade folder structure.
5. List development phases with goals, tasks, expected outputs.
6. Detail RAG pipeline: chunking, embedding, retrieval, strict grounding prompt.
7. Define API endpoints with example payloads.
8. Define error handling, thresholds, and fallback behavior.
9. Define testing strategy, include retrieval evaluation.
10. Define deployment plan with Docker and cloud options.
11. Define coding standards and logging rules.
12. Define Copilot usage guidelines.
13. Validate: all required sections present, no vague language, implementation-ready.
14. Save as PROJECT_PLAN.md.

## Quality Checklist
- Strict grounding and unknown handling explicit
- Dynamic knowledge updates included
- Context-aware retrieval and memory included
- Multi-format ingestion included
- Auth and logging included
- GitHub-ready structure included
- Thresholds for abstention defined
- Clean markdown with code blocks for structures

## Notes
- Keep plan practical and scoped. Avoid theory.
