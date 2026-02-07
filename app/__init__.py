"""
FixTrade — Intelligent Trading Assistant for the BVMT.

Application package root. This is a modular monolith using
hexagonal architecture (ports & adapters) with domain-driven design.

Bounded contexts:
    - trading: Price prediction, sentiment, anomaly detection, portfolio management.

Layers:
    - domain: Pure business logic, entities, ports (ABCs), errors.
    - application: Use cases, DTOs, orchestration.
    - infrastructure: Adapters (DB, ML, APIs) implementing domain ports.
    - interfaces: FastAPI routers, Pydantic schemas.
    - shared: Cross-cutting concerns (errors, security, logging).
"""
