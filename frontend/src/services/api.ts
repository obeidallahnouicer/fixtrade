const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api/v1";

function getAuthToken(): string | null {
  return localStorage.getItem("fixtrade_auth_token");
}

async function apiFetch(path: string, options: RequestInit = {}, includeAuth = false) {
  const headers = new Headers(options.headers || {});
  headers.set("Content-Type", "application/json");

  if (includeAuth) {
    const token = getAuthToken();
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const responseText = await response.text();
    let body: any = null;
    try {
      body = responseText ? JSON.parse(responseText) : null;
    } catch {
      body = null;
    }
    const detail = body?.detail;
    const validationMessage = Array.isArray(detail)
      ? detail
          .map((item: { loc?: Array<string | number>; msg?: string }) => {
            const field = item.loc?.[item.loc.length - 1];
            return field ? `${String(field)}: ${item.msg}` : item.msg;
          })
          .filter(Boolean)
          .join(". ")
      : null;
    const proxyMessage =
      response.status === 504
        ? "Le calcul est terminé côté serveur, mais le proxy a expiré en attendant l'IA locale."
        : null;
    const message =
      body?.error ||
      body?.message ||
      validationMessage ||
      detail ||
      proxyMessage ||
      `Request failed (${response.status})`;
    throw new Error(message);
  }

  return response.json();
}

export async function fetchPricePredictions(symbol: string) {
  return apiFetch(`/trading/predictions`, {
    method: "POST",
    body: JSON.stringify({ symbol, horizon_days: 5 }),
  });
}

export async function fetchSentiment(symbol: string) {
  return apiFetch(`/trading/sentiment`, {
    method: "POST",
    body: JSON.stringify({ symbol }),
  });
}

export async function fetchAnomalies(symbol: string) {
  return apiFetch(`/trading/anomalies`, {
    method: "POST",
    body: JSON.stringify({ symbol }),
  });
}

export async function fetchRecommendation(symbol: string) {
  return apiFetch(`/trading/recommendations`, {
    method: "POST",
    body: JSON.stringify({ symbol, portfolio_id: "00000000-0000-0000-0000-000000000000" }),
  });
}

export interface AuthUser {
  id: string;
  email: string;
  full_name: string | null;
  role: string;
  is_active: boolean;
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: AuthUser;
}

export interface DashboardBootstrapResponse {
  symbol: string;
  current_snapshot: {
    date: string;
    close: string;
    previous_close: string;
    volume: number;
    average_volume: number;
  } | null;
  historical_prices: Array<{
    date: string;
    close: string;
  }>;
  price_predictions: Array<{
    symbol: string;
    target_date: string;
    predicted_close: string;
    confidence_lower: string;
    confidence_upper: string;
  }>;
  sentiment: {
    symbol: string;
    date: string;
    score: string;
    sentiment: string;
    article_count: number;
  } | null;
  anomalies: Array<{
    id: string;
    symbol: string;
    detected_at: string;
    anomaly_type: string;
    severity: string;
    description: string;
  }>;
  recommendation: {
    symbol: string;
    action: "BUY" | "SELL" | "HOLD";
    confidence: string;
    reasoning: string;
  } | null;
  warnings: string[];
}

export interface MarketSnapshotResponse {
  symbol: string;
  date: string;
  close: string;
  previous_close: string;
  change: string;
  change_percent: string;
  volume: number;
  average_volume: number;
  sentiment_score: string;
  recommendation: "buy" | "sell" | "hold" | null;
  recommendation_confidence: string;
  anomaly_count: number;
}

export interface LoginPayload {
  email: string;
  password: string;
}

export interface RegisterPayload extends LoginPayload {
  full_name?: string | null;
}

export async function loginUser(payload: LoginPayload): Promise<AuthResponse> {
  return apiFetch(`/auth/login`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function registerUser(payload: RegisterPayload): Promise<AuthResponse> {
  return apiFetch(`/auth/register`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function fetchCurrentUser(): Promise<AuthUser> {
  return apiFetch(`/auth/me`, {
    method: "GET",
  }, true);
}

export async function fetchDashboardBootstrap(symbol: string): Promise<DashboardBootstrapResponse> {
  return apiFetch(`/dashboard/bootstrap?symbol=${encodeURIComponent(symbol)}`);
}

export async function fetchMarketUniverse(limit = 60): Promise<MarketSnapshotResponse[]> {
  const response = await apiFetch(`/dashboard/markets?limit=${limit}`);
  return response.markets || [];
}

export type PortfolioRiskProfile = "conservative" | "moderate" | "aggressive";

export interface PortfolioOptimizationResponse {
  risk_profile: PortfolioRiskProfile;
  company_count: number;
  investment_amount: number;
  assets: Array<{
    symbol: string;
    latest_price: number;
    weight: number;
    allocation_amount: number;
    shares: number;
    invested_amount: number;
    beta: number;
    covariance_with_market: number;
    volatility: number;
    capm_return: number;
  }>;
  metrics: {
    expected_return: number;
    volatility: number;
    variance: number;
    beta: number;
    market_return: number;
    risk_free_rate: number;
    invested_amount: number;
    cash_remaining: number;
  };
  efficient_frontier: Array<{
    expected_return: number;
    volatility: number;
    is_pvm: boolean;
  }>;
  methodology: {
    observations: number;
    trading_days_per_year: number;
    market_variance: number;
    market_risk_premium: number;
    minimum_weight: number;
    maximum_weight: number;
  };
  explanation: string;
  explanation_source: "openrouter" | "lm_studio" | "multi_agent" | "fallback";
  generated_at: string;
  warnings: string[];
}

export async function optimizePortfolio(payload: {
  risk_profile: PortfolioRiskProfile;
  company_count: number;
  investment_amount: number;
}): Promise<PortfolioOptimizationResponse> {
  return apiFetch("/portfolio/optimize", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
