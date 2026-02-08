"""
Secure HTTP headers middleware.

Adds security-related headers to every response:
- X-Content-Type-Options
- X-Frame-Options
- Referrer-Policy
- Content-Security-Policy
- X-XSS-Protection

No business logic. Pure cross-cutting concern.
"""

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

SECURE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Content-Security-Policy": "default-src 'self'",
    "X-XSS-Protection": "1; mode=block",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware that adds secure HTTP headers to every response.

    Prevents common web vulnerabilities by setting restrictive
    default headers on all outgoing responses.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request and add security headers to response."""
        response = await call_next(request)
        for header_name, header_value in SECURE_HEADERS.items():
            response.headers[header_name] = header_value
        return response
