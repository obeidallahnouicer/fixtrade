import React, { useMemo } from "react";
import { useStore } from "@/store/useStore";
import { useAuthStore } from "@/store/useAuthStore";
import { StockList } from "@/components/trading/StockList";
import { PredictionChart } from "@/components/trading/PredictionChart";
import { AnomalyList } from "@/components/trading/AnomalyList";
import { AICard } from "@/components/trading/AICard";
import { SentimentGauge } from "@/components/trading/SentimentGauge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { formatCurrency, formatPercentage, formatNumber } from "@/lib/utils";
import { Activity, Bell, Cpu, Menu, Search, Settings } from "lucide-react";

export function Dashboard() {
  const { stocks, selectedStockId, chartData, recommendation } = useStore();
  const { user, signOut } = useAuthStore();
  
  const selectedStock = useMemo(() => 
    stocks.find(s => s.symbol === selectedStockId) || stocks[0], 
  [stocks, selectedStockId]);

  React.useEffect(() => {
    useStore.getState().fetchStockData(selectedStockId);
  }, [selectedStockId]);

  return (
    <div className="min-h-screen bg-[#09090b] text-[#ededed] flex flex-col font-sans">
      {/* Top Navbar */}
      <header className="h-14 border-b border-[#27272a] bg-[#121214]/80 backdrop-blur flex items-center justify-between px-6 shrink-0 sticky top-0 z-10">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2 text-emerald-500 font-bold tracking-tight text-lg">
            <Activity fill="currentColor" size={18} />
            FixTrade
          </div>
          <nav className="hidden md:flex items-center gap-4 text-sm font-medium text-zinc-400">
            <span className="text-zinc-100 cursor-pointer">Dashboard</span>
            <span className="hover:text-zinc-100 transition-colors cursor-pointer">Portfolio</span>
            <span className="hover:text-zinc-100 transition-colors cursor-pointer">Markets</span>
            <span className="hover:text-zinc-100 transition-colors cursor-pointer">Screener</span>
          </nav>
        </div>
        
        <div className="flex items-center gap-4 text-zinc-400">
          <Search size={18} className="cursor-pointer hover:text-zinc-100 transition-colors" />
          <Bell size={18} className="cursor-pointer hover:text-zinc-100 transition-colors" />
          <Settings size={18} className="cursor-pointer hover:text-zinc-100 transition-colors" />
          <div className="hidden md:flex flex-col items-end mr-2">
            <span className="text-xs font-semibold text-zinc-100">{user?.full_name || user?.email || "Guest"}</span>
            <span className="text-[10px] uppercase tracking-[0.2em] text-zinc-500">{user?.role || "user"}</span>
          </div>
          <button
            type="button"
            onClick={signOut}
            className="px-3 py-1.5 rounded-full border border-zinc-700 bg-zinc-900 text-xs font-semibold text-zinc-200 hover:border-zinc-500 hover:text-white transition-colors"
          >
            Logout
          </button>
          <div className="w-8 h-8 rounded-full bg-zinc-800 border border-zinc-700 flex items-center justify-center">
            <span className="text-xs font-medium text-zinc-100">{(user?.email || "ON").slice(0, 2).toUpperCase()}</span>
          </div>
        </div>
      </header>

      {/* Ticker Tape */}
      <div className="h-8 bg-[#121214] border-b border-[#27272a] flex items-center overflow-hidden shrink-0 px-6 font-mono text-[11px] whitespace-nowrap">
        {stocks.map(s => (
          <div key={s.symbol} className="flex items-center gap-2 mr-8">
            <span className="text-zinc-400">{s.symbol}</span>
            <span className="text-zinc-100">{s.price.toFixed(2)}</span>
            <span className={s.change >= 0 ? "text-emerald-400" : "text-red-400"}>
              {s.change >= 0 ? "+" : ""}{s.changePercent.toFixed(2)}%
            </span>
          </div>
        ))}
      </div>

      {/* Main Grid */}
      <main className="flex-1 p-4 md:p-6 grid grid-cols-1 md:grid-cols-12 gap-6 overflow-hidden">
        
        {/* Left Col - Watchlist */}
        <div className="md:col-span-3 lg:col-span-2 hidden md:block border-r border-[#27272a] pr-6 overflow-y-auto">
          <StockList />
        </div>

        {/* Center Col - Main Chart inside */}
        <div className="md:col-span-9 lg:col-span-7 flex flex-col space-y-6 overflow-y-auto min-h-0 pb-12">
          
          <div className="flex flex-col space-y-6">
            <div className="flex items-start justify-between">
              <div className="flex flex-col">
                <div className="flex items-center gap-3">
                  <h1 className="text-3xl font-bold tracking-tight text-white">{selectedStock.symbol}</h1>
                  <Badge variant="default">{selectedStock.name}</Badge>
                </div>
                <div className="flex items-end gap-3 mt-2">
                  <span className="text-4xl font-mono tracking-tight font-medium">
                    {formatCurrency(selectedStock.price)}
                  </span>
                  <div className={`flex flex-col font-mono text-sm pb-1 ${selectedStock.change >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                    <span>{selectedStock.change >= 0 ? "+" : ""}{formatCurrency(selectedStock.change)}</span>
                    <span>({selectedStock.change >= 0 ? "+" : ""}{formatPercentage(selectedStock.changePercent)})</span>
                  </div>
                </div>
              </div>
              
              <div className="flex gap-4 text-right">
                <div className="flex flex-col">
                  <span className="text-xs text-zinc-500 uppercase font-semibold tracking-wide">Volume</span>
                  <span className="font-mono text-zinc-100">{formatNumber(selectedStock.volume)}</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-xs text-zinc-500 uppercase font-semibold tracking-wide">Avg Vol 20D</span>
                  <span className="font-mono text-zinc-100">{formatNumber(selectedStock.avgVolume)}</span>
                </div>
              </div>
            </div>

            <Card>
              <CardHeader className="pb-0 flex flex-row items-center justify-between border-b border-[#27272a]">
                <CardTitle className="text-sm font-semibold uppercase tracking-widest text-zinc-400 mb-3 block">
                  Ensemble Forecast (LSTM + XGBoost + Prophet)
                </CardTitle>
                <div className="flex gap-2 mb-3">
                  <Badge variant="neutral">1D</Badge>
                  <Badge variant="neutral">5D</Badge>
                  <Badge variant="success">CONF: 95%</Badge>
                </div>
              </CardHeader>
              <CardContent className="pt-2 px-0">
                <PredictionChart data={chartData} />
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Right Col - AI & Anomalies */}
        <div className="md:col-span-12 lg:col-span-3 flex flex-col space-y-6 overflow-y-auto">
          
          <AICard recommendation={recommendation || {
            symbol: selectedStock.symbol,
            action: "HOLD",
            confidence: 0,
            predictedReturn: 0,
            horizonDays: 0,
            reasoning: "Waiting for recommendation..."
          }} />

          <Card>
            <CardHeader className="pb-3 border-b border-[#27272a] mb-4">
              <CardTitle className="text-xs font-semibold uppercase text-zinc-400 tracking-widest">
                Market Sentiment
              </CardTitle>
            </CardHeader>
            <CardContent>
              <SentimentGauge score={selectedStock.sentimentScore} />
            </CardContent>
          </Card>

          <div className="flex flex-col">
            <h3 className="text-xs font-semibold uppercase tracking-widest text-zinc-400 mb-4 flex items-center gap-2">
              <Activity size={14} className="text-amber-500" />
              Live Anomalies
            </h3>
            <AnomalyList />
          </div>

        </div>

      </main>
    </div>
  );
}
