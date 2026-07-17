# FixTrade Microservices Thesis Roadmap

This roadmap is optimized for an end-of-studies thesis demo: it should look like a real microservices platform, show clear technical mastery, and stay achievable inside a student project timeline.

## First Principle

Do not start by splitting code randomly. Start by defining the service boundaries, the user journey, and the API contract that the frontend will consume.

The current codebase is a modular monolith with a separate frontend, a FastAPI backend, PostgreSQL, Redis, ETL/scraping, ML inference, and GenAI logic. The repo also explicitly says authentication is not implemented yet, so identity must be introduced early, not treated as a last-minute add-on.

## What To Do First

1. Freeze the demo scope.
   Decide the exact story you want to show in the defense: registration, login, dashboard, predictions, sentiment/explanations, portfolio view, and at least one background pipeline.

2. Define the target services.
   Keep the boundaries simple and explainable: frontend, auth, API gateway/BFF, ETL worker, ML inference, GenAI/explainability, portfolio/trading, and local infrastructure.

3. Decide how services talk to each other.
   Use synchronous HTTP for user-facing reads, async jobs for ETL/training, and keep the frontend talking only to the gateway.

4. Design data ownership.
   Keep Postgres local in Docker, but assign clear ownership per service. Avoid all services reading and writing the same tables directly.

5. Define security basics early.
   Plan JWT or session auth, registration, password hashing, role-based access control, rate limiting, security headers, and CORS rules before implementation begins.

## Recommended Build Order

### Phase 1: Foundation

- Create the architecture diagram.
- Define the service contracts and endpoints.
- Add auth and registration.
- Add a gateway/BFF layer.
- Make the frontend consume the new backend contract.

### Phase 2: High-Value Demo Services

- Extract ML inference into its own service.
- Extract GenAI/explainability into its own service.
- Keep the frontend focused on display and interaction.

### Phase 3: Background and Data Services

- Move ETL and scraping into a worker service.
- Separate training/retraining from inference.
- Add queue or scheduled-job behavior if needed for realism.

### Phase 4: Product Features

- Add portfolio persistence.
- Add user-specific recommendations.
- Add audit/logging and health endpoints.
- Polish the dashboard and flow for the thesis demo.

## Minimal Service Map

| Service | Responsibility |
|---|---|
| Frontend | UI, charts, auth screens, dashboards |
| Auth | Registration, login, tokens, roles |
| Gateway/BFF | Single entry point for the frontend |
| ETL Worker | Scraping, cleaning, ingestion, scheduled jobs |
| ML Service | Model scoring and prediction APIs |
| GenAI Service | Explanations, summaries, prompt orchestration |
| Portfolio Service | Virtual portfolio, trades, recommendations |
| Postgres | Local persistent storage in Docker |
| Redis | Cache, job coordination, lightweight state |

## Thesis-Friendly Security Baseline

- Hash passwords with a strong password hashing algorithm.
- Use access tokens and refresh tokens, or secure cookie sessions.
- Add role checks for admin and regular users.
- Add rate limiting for login and prediction endpoints.
- Add security headers and strict CORS.
- Never expose raw secrets in logs.

## What Not To Do First

- Do not split everything into tiny services immediately.
- Do not put all services on the same shared database schema.
- Do not add Kubernetes before the service boundaries are stable.
- Do not build training infrastructure before auth and the frontend contract are settled.

## Best Thesis Narrative

The clean story is: a modular monolith was decomposed into microservices around bounded contexts, the frontend was connected through a gateway, identity was centralized, ML and GenAI were isolated as specialized services, and background ETL was separated from user-facing requests.

That gives you a strong architecture story without creating an overcomplicated system.