import { create } from "zustand";
import {
  StockSnapshot,
  PricePoint,
  AnomalyAlert,
  AIRecommendation,
} from "@/types/trading";
import {
  fetchDashboardBootstrap,
  fetchMarketUniverse,
} from "@/services/api";

const FALLBACK_STOCK: StockSnapshot = {
  symbol: "BIAT",
  name: "BIAT",
  price: 0,
  change: 0,
  changePercent: 0,
  volume: 0,
  avgVolume: 0,
  sentimentScore: 0,
};

interface TradingStore {
  selectedStockId: string;
  setSelectedStockId: (id: string) => void;
  stocks: StockSnapshot[];
  chartData: PricePoint[];
  anomalies: AnomalyAlert[];
  recommendation: AIRecommendation | null;
  loading: boolean;
  marketsLoading: boolean;
  error: string | null;
  warnings: string[];
  fetchMarkets: () => Promise<void>;
  fetchStockData: (symbol: string) => Promise<void>;
}

export const useStore = create<TradingStore>((set) => ({
  selectedStockId: FALLBACK_STOCK.symbol,
  setSelectedStockId: (id) => set({ selectedStockId: id }),
  stocks: [FALLBACK_STOCK],
  chartData: [],
  anomalies: [],
  recommendation: null,
  loading: false,
  marketsLoading: false,
  error: null,
  warnings: [],

  fetchMarkets: async () => {
    set({ marketsLoading: true, error: null });
    try {
      const markets = await fetchMarketUniverse();
      const stocks: StockSnapshot[] = markets.map((market) => ({
        symbol: market.symbol,
        name: market.symbol,
        price: Number(market.close),
        change: Number(market.change),
        changePercent: Number(market.change_percent),
        volume: market.volume,
        avgVolume: market.average_volume,
        sentimentScore: Number(market.sentiment_score),
        recommendation: market.recommendation?.toUpperCase() as
          | StockSnapshot["recommendation"]
          | undefined,
        recommendationConfidence: Number(market.recommendation_confidence),
        anomalyCount: market.anomaly_count,
        marketDate: market.date,
      }));

      set((state) => ({
        stocks: stocks.length > 0 ? stocks : state.stocks,
        selectedStockId: stocks.some(
          (stock) => stock.symbol === state.selectedStockId
        )
          ? state.selectedStockId
          : stocks[0]?.symbol || state.selectedStockId,
        marketsLoading: false,
      }));
    } catch (error) {
      set({
        marketsLoading: false,
        error: error instanceof Error ? error.message : "Unable to load markets",
      });
    }
  },

  fetchStockData: async (symbol: string) => {
    set({ loading: true, error: null, warnings: [] });
    try {
      const data = await fetchDashboardBootstrap(symbol);

      const historicalSeries = (data.historical_prices || []).map((point) => ({
        date: point.date,
        historicalPrice: Number(point.close),
      }));

      const forecastByDate = new Map<string, PricePoint>();
      (data.price_predictions || []).forEach((point) => {
        forecastByDate.set(point.target_date, {
          date: point.target_date,
          predictedPrice: Number(point.predicted_close),
          confLower: Number(point.confidence_lower),
          confUpper: Number(point.confidence_upper),
        });
      });

      const historicalByDate = new Map<string, PricePoint>();
      historicalSeries.forEach((point) => historicalByDate.set(point.date, point));

      const chartData: PricePoint[] = [
        ...historicalByDate.values(),
        ...[...forecastByDate.entries()]
          .filter(([dateKey]) => !historicalByDate.has(dateKey))
          .map(([, point]) => point),
      ].sort((left, right) => left.date.localeCompare(right.date));

      const anomalies: AnomalyAlert[] = data.anomalies.map((anomaly) => ({
        id: anomaly.id,
        symbol: anomaly.symbol,
        type: anomaly.anomaly_type,
        severity: Number(anomaly.severity),
        description: anomaly.description,
        detectedAt: anomaly.detected_at,
      }));

      const latest = data.current_snapshot
        ? Number(data.current_snapshot.close)
        : historicalSeries[historicalSeries.length - 1]?.historicalPrice ?? 0;
      const finalForecast = [...(data.price_predictions || [])]
        .sort((left, right) => left.target_date.localeCompare(right.target_date))
        .at(-1);
      const finalForecastPrice = finalForecast
        ? Number(finalForecast.predicted_close)
        : latest;
      const predictedReturn =
        latest > 0 && Number.isFinite(finalForecastPrice)
          ? ((finalForecastPrice - latest) / latest) * 100
          : 0;

      const recommendation = data.recommendation
        ? ({
            symbol: data.recommendation.symbol,
            action:
              data.recommendation.action.toUpperCase() as AIRecommendation["action"],
            confidence: Number(data.recommendation.confidence),
            reasoning: data.recommendation.reasoning,
            predictedReturn,
            horizonDays: data.price_predictions.length || 5,
          } as AIRecommendation)
        : null;

      const previous = data.current_snapshot
        ? Number(data.current_snapshot.previous_close)
        : historicalSeries[historicalSeries.length - 2]?.historicalPrice ?? latest;
      const change = latest - previous;

      set((state) => ({
        stocks: state.stocks.map((stock) =>
          stock.symbol === symbol
            ? {
                ...stock,
                price: latest,
                change,
                changePercent: previous ? (change / previous) * 100 : 0,
                volume: data.current_snapshot?.volume ?? stock.volume,
                avgVolume:
                  data.current_snapshot?.average_volume ?? stock.avgVolume,
                sentimentScore: Number(
                  data.sentiment?.score ?? stock.sentimentScore
                ),
                recommendation: recommendation?.action ?? stock.recommendation,
                recommendationConfidence:
                  recommendation?.confidence ?? stock.recommendationConfidence,
                anomalyCount: anomalies.length,
              }
            : stock
        ),
        chartData,
        anomalies,
        recommendation,
        warnings: data.warnings || [],
        loading: false,
      }));
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : "Request failed",
        loading: false,
      });
    }
  },
}));
