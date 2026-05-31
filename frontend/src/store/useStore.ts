import { create } from "zustand";
import { StockSnapshot, PricePoint, AnomalyAlert, AIRecommendation } from "@/types/trading";
import { fetchPricePredictions, fetchSentiment, fetchAnomalies, fetchRecommendation } from "@/services/api";

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
      const [predsRes, sentimentRes, anomaliesRes, recRes] = await Promise.allSettled([
        fetchPricePredictions(symbol),
        fetchSentiment(symbol),
        fetchAnomalies(symbol),
        fetchRecommendation(symbol)
      ]);
      
      let newChartData: PricePoint[] = [];
      if (predsRes.status === "fulfilled" && predsRes.value.predictions) {
        newChartData = predsRes.value.predictions.map((p: any) => ({
             date: p.target_date, 
             predictedPrice: p.predicted_close, 
             confLower: p.confidence_lower, 
             confUpper: p.confidence_upper 
        }));
      }

      let newAnomalies: AnomalyAlert[] = [];
      if (anomaliesRes.status === "fulfilled" && anomaliesRes.value.anomalies) {
         newAnomalies = anomaliesRes.value.anomalies.map((a: any) => ({
             id: a.id,
             symbol: a.symbol,
             type: a.anomaly_type,
             severity: a.severity,
             description: a.description,
             detectedAt: a.detected_at
         }));
      }

      let newRecommendation = null;
      if (recRes.status === "fulfilled") {
         newRecommendation = recRes.value as AIRecommendation;
      }

      // Update static list with sentiment score if we got it
      if (sentimentRes.status === "fulfilled" && sentimentRes.value.score !== undefined) {
         set((state) => ({
           stocks: state.stocks.map(s => s.symbol === symbol ? {
             ...s,
             sentimentScore: sentimentRes.value.score
           } : s)
         }));
      }

      let recommendationValue = null;
      if (newRecommendation?.action) {
        recommendationValue = newRecommendation;
      }
      
      set({ 
        chartData: newChartData, 
        anomalies: newAnomalies, 
        recommendation: recommendationValue,
        loading: false 
      });
      
    } catch (e: any) {
      set({ error: e.message, loading: false });
    }
  }
}));
