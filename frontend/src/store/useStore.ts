import { create } from "zustand";
import { StockSnapshot, PricePoint, AnomalyAlert, AIRecommendation } from "@/types/trading";
import { fetchDashboardBootstrap } from "@/services/api";

const STATIC_STOCKS: StockSnapshot[] = [
  { symbol: "BIAT", name: "Banque Internationale Arabe de Tunisie", price: 0, change: 0, changePercent: 0, volume: 0, avgVolume: 0, sentimentScore: 0 },
  { symbol: "SFBT", name: "Société de Fabrication des Boissons de Tunisie", price: 0, change: 0, changePercent: 0, volume: 0, avgVolume: 0, sentimentScore: 0 },
  { symbol: "BT", name: "Banque de Tunisie", price: 0, change: 0, changePercent: 0, volume: 0, avgVolume: 0, sentimentScore: 0 }
];

interface TradingStore {
  selectedStockId: string;
  setSelectedStockId: (id: string) => void;
  stocks: StockSnapshot[];
  
  // API Data
  chartData: PricePoint[];
  anomalies: AnomalyAlert[];
  recommendation: AIRecommendation | null;
  loading: boolean;
  error: string | null;
  fetchStockData: (symbol: string) => Promise<void>;
}

export const useStore = create<TradingStore>((set, get) => ({
  selectedStockId: STATIC_STOCKS[0].symbol,
  setSelectedStockId: (id) => {
    set({ selectedStockId: id });
    get().fetchStockData(id);
  },
  stocks: STATIC_STOCKS,
  chartData: [],
  anomalies: [],
  recommendation: null,
  loading: false,
  error: null,
  
  fetchStockData: async (symbol: string) => {
    set({ loading: true, error: null });
    try {
      const data = await fetchDashboardBootstrap(symbol);

      const historicalSeries = (data.historical_prices || []).map((p) => ({
        date: p.date,
        historicalPrice: Number(p.close),
      }));

      const forecastByDate = new Map<string, PricePoint>();
      (data.price_predictions || []).forEach((p) => {
        forecastByDate.set(p.target_date, {
          date: p.target_date,
          predictedPrice: Number(p.predicted_close),
          confLower: Number(p.confidence_lower),
          confUpper: Number(p.confidence_upper),
        });
      });

      const historicalByDate = new Map<string, PricePoint>();
      historicalSeries.forEach((point) => {
        historicalByDate.set(point.date, point);
      });

      const newChartData: PricePoint[] = [
        ...historicalByDate.values(),
        ...[...forecastByDate.entries()]
          .filter(([dateKey]) => !historicalByDate.has(dateKey))
          .map(([, point]) => point),
      ].sort((left, right) => left.date.localeCompare(right.date));

      const newAnomalies: AnomalyAlert[] = data.anomalies.map((a) => ({
        id: a.id,
        symbol: a.symbol,
        type: a.anomaly_type,
        severity: Number(a.severity),
        description: a.description,
        detectedAt: a.detected_at,
      }));

      const newRecommendation = data.recommendation
        ? ({
            symbol: data.recommendation.symbol,
            action: data.recommendation.action,
            confidence: Number(data.recommendation.confidence),
            reasoning: data.recommendation.reasoning,
            predictedReturn: 0,
            horizonDays: 5,
          } as AIRecommendation)
        : null;

      if (data.sentiment?.score !== undefined) {
        set((state) => ({
          stocks: state.stocks.map((s) =>
            s.symbol === symbol
              ? {
                  ...s,
                  sentimentScore: Number(data.sentiment?.score ?? s.sentimentScore),
                }
              : s,
          ),
        }));
      }

      set({ 
        chartData: newChartData, 
        anomalies: newAnomalies, 
        recommendation: newRecommendation,
        loading: false 
      });
      
    } catch (e: any) {
      set({ error: e.message, loading: false });
    }
  }
}));
