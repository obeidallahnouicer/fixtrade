export async function fetchPricePredictions(symbol: string) {
  const response = await fetch(`/api/v1/trading/predictions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol, horizon_days: 5 }),
  });
  if (!response.ok) throw new Error("Failed to fetch predictions");
  return response.json();
}

export async function fetchSentiment(symbol: string) {
  const response = await fetch(`/api/v1/trading/sentiment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol }),
  });
  if (!response.ok) throw new Error("Failed to fetch sentiment");
  return response.json();
}

export async function fetchAnomalies(symbol: string) {
  const response = await fetch(`/api/v1/trading/anomalies`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol }),
  });
  if (!response.ok) throw new Error("Failed to fetch anomalies");
  return response.json();
}

export async function fetchRecommendation(symbol: string) {
  const response = await fetch(`/api/v1/trading/recommendations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol, portfolio_id: "00000000-0000-0000-0000-000000000000" }),
  });
  if (!response.ok) throw new Error("Failed to fetch recommendation");
  return response.json();
}
