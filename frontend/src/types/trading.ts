export interface StockSnapshot {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  avgVolume: number;
  sentimentScore: number;
  recommendation?: "BUY" | "SELL" | "HOLD";
  recommendationConfidence?: number;
  anomalyCount?: number;
  marketDate?: string;
}

export interface PricePoint {
  date: string;
  historicalPrice?: number;
  predictedPrice?: number;
  confLower?: number;
  confUpper?: number;
}

export interface AnomalyAlert {
  id: string;
  symbol: string;
  type: string;
  severity: number;
  description: string;
  detectedAt: string;
}

export interface AIRecommendation {
  symbol: string;
  action: "BUY" | "SELL" | "HOLD";
  confidence: number;
  reasoning: string;
  predictedReturn: number;
  horizonDays: number;
}
