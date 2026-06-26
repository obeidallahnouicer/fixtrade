# AGENTS.md

## Project

FixTrade is a BVMT trading intelligence platform for quantitative analysis, market prediction, anomaly detection, sentiment analysis, and GenAI-based portfolio recommendation.

The project is used for a Master thesis. Code changes must preserve technical clarity, architecture consistency, and documentation quality.

## Main stack

- Backend: FastAPI, Python, PostgreSQL, Redis
- Frontend: React 19, Vite, TypeScript/JavaScript, Tailwind CSS
- Data: BVMT OHLCV data, Managers.tn news, extracted PDFs
- ML: Prophet, XGBoost, LSTM, anomaly detection, risk metrics
- GenAI: portfolio recommendation, explainability, investment reasoning
- DevOps: Docker / Docker Compose where applicable

## Architecture rules

- Respect the existing architecture.
- Do not rewrite the whole project unless explicitly requested.
- Keep backend logic separated by domain/service/router/schema layers where possible.
- Keep frontend components reusable and avoid putting business logic inside UI components.
- Follow Bronze / Silver / Gold data-pipeline logic when relevant.
- Keep ML code reproducible and documented.

## Coding rules

- Do not invent fake metrics, fake datasets, or fake thesis results.
- Do not remove existing functionality without explaining why.
- Prefer readable, maintainable code over clever code.
- Keep naming consistent with the existing project.
- Avoid large unrequested refactors.
- Inspect relevant files before editing.
- After editing, summarize changed files and why each change was made.

## Backend rules

- Use FastAPI patterns cleanly.
- Keep endpoint logic thin when possible.
- Validate inputs with Pydantic schemas.
- Handle errors explicitly.
- Do not expose secrets, API keys, tokens, or credentials.
- Avoid blocking operations inside async routes unless unavoidable.
- Use clear HTTP status codes.

## Frontend rules

- Preserve the current visual identity unless asked to redesign.
- Use existing components and styling conventions.
- Keep pages responsive.
- Avoid unnecessary dependencies.
- Do not break routes or navigation.

## Data and ML rules

- Maintain separation between raw data, cleaned data, features, predictions, and explanations.
- Never leak future data into training or evaluation.
- Explain model changes in thesis-friendly language.
- For anomalies, distinguish between statistical anomaly, sentiment-price divergence, and data-quality issue.

## GenAI rules

- Recommendations must be explainable, cautious, and grounded in available data.
- Do not present generated portfolio recommendations as guaranteed financial advice.
- Always include risk reasoning.
- Prefer structured output.

## Documentation rules

When asked to document, write in formal French unless the user asks otherwise.

For thesis documentation:

- Be detailed.
- Explain why each technology was used.
- Include architecture reasoning.
- Include diagrams when useful.
- Avoid repetition.
- Keep claims realistic and defensible.

## Work style

1. Inspect files first.
2. Identify the minimal safe change.
3. Apply the change.
4. Run or suggest relevant checks.
5. Summarize what changed.
6. Mention risks or assumptions.

## Do not do

- Do not fabricate results.
- Do not overwrite large files blindly.
- Do not delete documentation unless requested.
- Do not change database schemas without explaining migration impact.
- Do not introduce unnecessary frameworks.
