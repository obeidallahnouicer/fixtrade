0. Project Philosophy (MANDATORY)

This project is an MVP.
Clarity > Cleverness. Predictability > Abstraction. Explicit > Implicit.

Copilot must:

Generate boring, readable, predictable code

Prefer simple control flow

Avoid premature optimization

Avoid over-engineering

If a solution is not easily explainable to a junior developer in 2 minutes, it is invalid.

1. Code Quality Rules (NON-NEGOTIABLE)
File & Function Limits

Max 200 lines per file

No function should have more than one responsibility

No class should have more than one public method

No God objects

No utility / helper dumping grounds

If logic grows:
→ split into use cases, services, or adapters

Cognitive Complexity

Avoid nested if/else deeper than 2 levels

Prefer early returns

Prefer small pure functions

No complex conditionals inline (extract intent-revealing functions)

Bad:

if a and (b or c) and not d:


Good:

if is_valid_trade(signal):

Dependencies

No unnecessary dependencies

Every dependency must:

Solve a real problem

Be widely used and maintained

Be documented in requirements.txt

If Python standard library can do it → use it

2. Architecture Rules
Architecture Style

Modular Monolith

Hexagonal Architecture (Ports & Adapters)

No microservices

No circular dependencies

Layering (STRICT)
domain/
application/
infrastructure/
interfaces/

Dependency Direction (ENFORCED)
interfaces → application → domain
infrastructure → application → domain
domain → NOTHING


The domain must never import FastAPI, SQLAlchemy, HTTP, or external libs.

Domain Layer

Pure business logic

Entities, Value Objects, Domain Services

No framework imports

No IO

No side effects

Application Layer

Use cases (one class = one use case)

Orchestration logic only

Calls domain + ports

No HTTP / DB logic

Infrastructure Layer

DB repositories

External APIs

ML models

Logging implementations

Implements ports defined in application layer.

Interfaces Layer

FastAPI routes

Request/response DTOs

Input validation

Error mapping

No business logic allowed here.

3. FastAPI Rules

One router per bounded context

Routes must call use cases, not services directly

No logic inside endpoints beyond:

parsing

validation

calling a use case

returning response

Bad:

@app.post("/trade")
def trade():
    if price > x:
        ...


Good:

@app.post("/trade")
def trade(cmd: TradeCommand):
    return execute_trade_use_case(cmd)

4. Error Handling (MANDATORY)
Principles

No silent failures

No generic Exception catching

No string-based error handling

Rules

Define custom exception types

Raise domain-specific errors

Map errors to HTTP responses at the interface layer

Example:

class InsufficientLiquidityError(Exception):
    pass


FastAPI:

@app.exception_handler(InsufficientLiquidityError)


Never expose stack traces or internal details to the client.

5. Inheritance & Composition
Inheritance

Use only when there is a true “is-a” relationship

Prefer composition over inheritance

No deep inheritance trees (max depth: 2)

Allowed:

Abstract base classes for ports

Shared domain behavior

Not allowed:

Inheritance for code reuse convenience

6. Documentation Rules
Mandatory Documentation

Every module has a docstring explaining its purpose

Every public class explains why it exists

Every use case explains:

input

output

side effects

failure cases

Code should read like a technical narrative.

7. Naming Rules

Names must express intent, not implementation

Avoid abbreviations

No single-letter variables outside loops

Use business language (ubiquitous language)

Bad:

def calc(x):


Good:

def calculate_liquidity_score():

8. Testing (MVP Level)

Focus on use case tests

Domain logic must be testable without DB or FastAPI

No snapshot tests

No brittle mocks

Rule:

If it’s important, it must be testable.

9. Logging & Observability

Structured logging only

Log at use-case boundaries

Never log sensitive data

Logging must not change program behavior

10. What NOT to Do

No magic

No metaprogramming

No decorators hiding logic

No “smart” abstractions

No global state

No monkey patching

11. Copilot-Specific Instructions

When generating code, Copilot must:

Follow this architecture strictly

Ask for clarification if a rule conflict arises

Prefer explicit code over abstractions

Generate code that a human can reason about

If unsure:
→ generate the simplest correct solution

12. MVP Reminder

This is an MVP.

Solve real problems

Defer optimizations

Avoid overengineering

Make future refactors easy

Perfect code that ships late is worse than good code that ships.

13. Security Rules (MANDATORY — AUTH OUT OF SCOPE)
Security Philosophy

Even without authentication, the application must be:

Safe by default

Fail-closed

Resilient to abuse

Defensive against malformed input

Security is not optional and must not be deferred.

13.1 Input Validation & Injection Prevention
General Rules

Never trust user input

All inputs must be:

Validated

Typed

Bounded

FastAPI / Pydantic

Use Pydantic models for all inputs

No raw dict, Any, or untyped JSON

Enforce:

min/max length

numeric bounds

regex where applicable

Bad:

def predict(payload: dict):


Good:

class PredictionRequest(BaseModel):
    symbol: constr(min_length=2, max_length=10)
    horizon: conint(ge=1, le=5)

SQL Injection

Never build SQL with string concatenation

Only use:

SQLAlchemy ORM

Parameterized queries

No raw SQL unless absolutely required (and documented)

Command / Code Injection

Never execute:

user-provided strings

dynamic imports

eval, exec, compile

These are forbidden.

13.2 Denial-of-Service (DoS) Protection
Rate Limiting (REQUIRED)

Apply rate limiting on all public endpoints

Use:

slowapi

or equivalent middleware

Rules:

Global default limit

Stricter limits for:

compute-heavy endpoints

ML inference

anomaly detection

Example policy:

60 req/min general

10 req/min heavy endpoints

Payload Size Limits

Enforce max request size

Reject large payloads early

No unbounded lists or arrays

Timeouts

All external calls must have:

timeouts

retries (bounded)

No infinite waits

13.3 Resource Protection
CPU & Memory Safety

No unbounded loops

No recursive logic

Batch ML inference only if explicitly controlled

Cap dataframe sizes and rows processed per request

Example:

MAX_ROWS = 5000

Concurrency Safety

Avoid global mutable state

Use dependency injection properly

No shared in-memory caches without locks or guarantees

13.4 Error Handling & Information Disclosure
Error Responses

Never leak:

stack traces

SQL errors

internal paths

User-facing errors must be:

generic

consistent

mapped explicitly

Example:

{
  "error": "Invalid request parameters"
}

Logging

Log errors internally with context

Never log:

request bodies

secrets

raw payloads

13.5 HTTP & API Hardening
Headers

Enforce secure headers:

X-Content-Type-Options

X-Frame-Options

Referrer-Policy

Content-Security-Policy (basic)

Methods

Disable unused HTTP methods

No auto-generated admin endpoints

No open debug routes

13.6 File & Path Safety (If Applicable)

Never accept raw file paths from users

Whitelist allowed file types

Store files outside execution paths

Sanitize filenames

13.7 Dependency Security

Pin dependency versions

Avoid unmaintained libraries

No wildcard installs

Document why each dependency exists

13.8 ML-Specific Security

Validate ML inputs strictly

Cap inference time

Protect against:

adversarial large inputs

model abuse

No dynamic model loading from user input

13.9 Security Testing (MVP-Level)

Add basic tests for:

invalid input

oversized payloads

rate limit enforcement

Test failure paths, not only success

13.10 Explicitly Out of Scope

Authentication

Authorization

User management

⚠️ These must not be partially implemented.

13.11 Copilot Security Instructions

When generating code, Copilot must:

Assume all endpoints are publicly exposed

Apply rate limiting by default

Validate all inputs

Fail safely and explicitly

Never trade security for convenience

If a feature cannot be implemented securely:
→ do not generate it