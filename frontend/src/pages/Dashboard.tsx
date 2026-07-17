import React, { useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  Bell,
  BriefcaseBusiness,
  Filter,
  Search,
  Settings,
} from "lucide-react";
import { useStore } from "@/store/useStore";
import { useAuthStore } from "@/store/useAuthStore";
import { StockList } from "@/components/trading/StockList";
import { PredictionChart } from "@/components/trading/PredictionChart";
import { AnomalyList } from "@/components/trading/AnomalyList";
import { AICard } from "@/components/trading/AICard";
import { SentimentGauge } from "@/components/trading/SentimentGauge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { StockSnapshot } from "@/types/trading";
import { formatCurrency, formatPercentage, formatNumber } from "@/lib/utils";
import {
  optimizePortfolio,
  PortfolioOptimizationResponse,
  PortfolioRiskProfile,
} from "@/services/api";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type ViewName = "dashboard" | "portfolio" | "markets" | "screener";

const NAV_ITEMS: Array<{ id: ViewName; label: string }> = [
  { id: "dashboard", label: "Dashboard" },
  { id: "portfolio", label: "Portfolio" },
  { id: "markets", label: "Markets" },
  { id: "screener", label: "Screener" },
];


export function Dashboard() {
  const {
    stocks,
    selectedStockId,
    chartData,
    recommendation,
    loading,
    marketsLoading,
    error,
    warnings,
    fetchMarkets,
    setSelectedStockId,
  } = useStore();
  const { user, signOut } = useAuthStore();
  const [activeView, setActiveView] = useState<ViewName>("dashboard");

  const selectedStock = useMemo(
    () => stocks.find((stock) => stock.symbol === selectedStockId) || stocks[0],
    [stocks, selectedStockId]
  );

  React.useEffect(() => {
    void fetchMarkets();
  }, [fetchMarkets]);

  React.useEffect(() => {
    if (selectedStockId) {
      void useStore.getState().fetchStockData(selectedStockId);
    }
  }, [selectedStockId]);

  function openStock(symbol: string) {
    setSelectedStockId(symbol);
    setActiveView("dashboard");
  }

  return (
    <div className="min-h-screen bg-[#09090b] text-[#ededed] flex flex-col font-sans">
      <header className="h-14 border-b border-[#27272a] bg-[#121214]/90 backdrop-blur flex items-center justify-between px-4 md:px-6 shrink-0 sticky top-0 z-20">
        <div className="flex items-center gap-5">
          <button
            type="button"
            onClick={() => setActiveView("dashboard")}
            className="flex items-center gap-2 text-emerald-500 font-bold tracking-tight text-lg"
          >
            <Activity fill="currentColor" size={18} />
            FixTrade
          </button>
          <nav className="hidden md:flex items-center gap-1 text-sm font-medium">
            {NAV_ITEMS.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setActiveView(item.id)}
                className={`rounded-md px-3 py-2 transition-colors ${
                  activeView === item.id
                    ? "bg-zinc-800 text-white"
                    : "text-zinc-400 hover:text-zinc-100"
                }`}
              >
                {item.label}
              </button>
            ))}
          </nav>
        </div>

        <div className="flex items-center gap-3 text-zinc-400">
          <button
            type="button"
            onClick={() => setActiveView("screener")}
            title="Open screener"
            className="hover:text-zinc-100"
          >
            <Search size={18} />
          </button>
          <Bell size={18} />
          <Settings size={18} />
          <div className="hidden lg:flex flex-col items-end">
            <span className="text-xs font-semibold text-zinc-100">
              {user?.full_name || user?.email || "Guest"}
            </span>
            <span className="text-[10px] uppercase tracking-[0.2em] text-zinc-500">
              {user?.role || "user"}
            </span>
          </div>
          <button
            type="button"
            onClick={signOut}
            className="px-3 py-1.5 rounded-full border border-zinc-700 bg-zinc-900 text-xs font-semibold text-zinc-200 hover:border-zinc-500"
          >
            Logout
          </button>
        </div>
      </header>

      <div className="md:hidden grid grid-cols-4 border-b border-zinc-800 bg-zinc-950">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            type="button"
            onClick={() => setActiveView(item.id)}
            className={`px-2 py-3 text-xs ${
              activeView === item.id ? "text-emerald-400" : "text-zinc-500"
            }`}
          >
            {item.label}
          </button>
        ))}
      </div>

      <div className="h-8 bg-[#121214] border-b border-[#27272a] flex items-center overflow-hidden shrink-0 px-6 font-mono text-[11px] whitespace-nowrap">
        {marketsLoading && <span className="text-zinc-500">Loading BVMT universe…</span>}
        {!marketsLoading &&
          stocks.slice(0, 14).map((stock) => (
            <button
              type="button"
              key={stock.symbol}
              onClick={() => openStock(stock.symbol)}
              className="flex items-center gap-2 mr-8 hover:opacity-80"
            >
              <span className="text-zinc-400">{stock.symbol}</span>
              <span className="text-zinc-100">{stock.price.toFixed(3)}</span>
              <span
                className={
                  stock.change >= 0 ? "text-emerald-400" : "text-red-400"
                }
              >
                {stock.change >= 0 ? "+" : ""}
                {stock.changePercent.toFixed(2)}%
              </span>
            </button>
          ))}
      </div>

      {activeView === "dashboard" && selectedStock && (
        <DashboardView
          stock={selectedStock}
          chartData={chartData}
          recommendation={recommendation}
          loading={loading}
          error={error}
          warnings={warnings}
        />
      )}
      {activeView === "markets" && (
        <MarketsView stocks={stocks} onOpenStock={openStock} />
      )}
      {activeView === "screener" && (
        <ScreenerView stocks={stocks} onOpenStock={openStock} />
      )}
      {activeView === "portfolio" && (
        <PortfolioView stocks={stocks} onOpenStock={openStock} />
      )}
    </div>
  );
}

function DashboardView({
  stock,
  chartData,
  recommendation,
  loading,
  error,
  warnings,
}: {
  stock: StockSnapshot;
  chartData: ReturnType<typeof useStore.getState>["chartData"];
  recommendation: ReturnType<typeof useStore.getState>["recommendation"];
  loading: boolean;
  error: string | null;
  warnings: string[];
}) {
  return (
    <main className="flex-1 p-4 md:p-6 grid grid-cols-1 md:grid-cols-12 gap-6">
      <aside className="md:col-span-3 lg:col-span-2 hidden md:block border-r border-[#27272a] pr-5 max-h-[calc(100vh-7rem)] overflow-y-auto">
        <StockList />
      </aside>

      <section className="md:col-span-9 lg:col-span-7 space-y-6 min-w-0">
        {(loading || error) && (
          <div
            className={`rounded-lg border px-4 py-3 text-sm ${
              error
                ? "border-amber-500/30 bg-amber-500/10 text-amber-200"
                : "border-zinc-800 bg-zinc-900 text-zinc-400"
            }`}
          >
            {error ? `Dashboard API unavailable: ${error}` : `Loading ${stock.symbol}…`}
          </div>
        )}
        {!loading && !error && warnings.length > 0 && (
          <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 px-4 py-3 text-xs text-amber-200/80">
            {warnings.join(" · ")}
          </div>
        )}

        <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-5">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-3xl font-bold tracking-tight text-white">
                {stock.symbol}
              </h1>
              <Badge variant="default">BVMT</Badge>
            </div>
            <div className="flex items-end gap-3 mt-2">
              <span className="text-4xl font-mono tracking-tight font-medium">
                {formatCurrency(stock.price)}
              </span>
              <div
                className={`font-mono text-sm pb-1 ${
                  stock.change >= 0 ? "text-emerald-400" : "text-red-400"
                }`}
              >
                <div>
                  {stock.change >= 0 ? "+" : ""}
                  {formatCurrency(stock.change)}
                </div>
                <div>
                  ({stock.change >= 0 ? "+" : ""}
                  {formatPercentage(stock.changePercent)})
                </div>
              </div>
            </div>
          </div>
          <div className="flex gap-5 text-right">
            <Metric label="Volume" value={formatNumber(stock.volume)} />
            <Metric label="Avg Vol 20D" value={formatNumber(stock.avgVolume)} />
          </div>
        </div>

        <Card>
          <CardHeader className="pb-0 flex flex-row items-center justify-between border-b border-[#27272a]">
            <CardTitle className="text-sm font-semibold uppercase tracking-widest text-zinc-400 mb-3">
              Persisted market forecast
            </CardTitle>
            <Badge variant="success">5 sessions</Badge>
          </CardHeader>
          <CardContent className="pt-2 px-0">
            <PredictionChart data={chartData} />
          </CardContent>
        </Card>
      </section>

      <aside className="md:col-span-12 lg:col-span-3 space-y-6">
        <AICard
          recommendation={
            recommendation || {
              symbol: stock.symbol,
              action: "HOLD",
              confidence: 0,
              predictedReturn: 0,
              horizonDays: 5,
              reasoning: "Waiting for the automated recommendation pipeline.",
            }
          }
        />
        <Card>
          <CardHeader className="pb-3 border-b border-[#27272a] mb-4">
            <CardTitle className="text-xs font-semibold uppercase text-zinc-400 tracking-widest">
              Market sentiment
            </CardTitle>
          </CardHeader>
          <CardContent>
            <SentimentGauge score={stock.sentimentScore} />
          </CardContent>
        </Card>
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-widest text-zinc-400 mb-4 flex items-center gap-2">
            <Activity size={14} className="text-amber-500" />
            Detected anomalies
          </h3>
          <AnomalyList />
        </div>
      </aside>
    </main>
  );
}

function MarketsView({
  stocks,
  onOpenStock,
}: {
  stocks: StockSnapshot[];
  onOpenStock: (symbol: string) => void;
}) {
  return (
    <Workspace title="Markets" subtitle={`${stocks.length} active BVMT instruments`}>
      <MarketTable stocks={stocks} onOpenStock={onOpenStock} />
    </Workspace>
  );
}

function ScreenerView({
  stocks,
  onOpenStock,
}: {
  stocks: StockSnapshot[];
  onOpenStock: (symbol: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [signal, setSignal] = useState("ALL");
  const [minimumVolume, setMinimumVolume] = useState(0);

  const filtered = useMemo(
    () =>
      stocks.filter(
        (stock) =>
          stock.symbol.toLowerCase().includes(query.toLowerCase()) &&
          (signal === "ALL" || stock.recommendation === signal) &&
          stock.avgVolume >= minimumVolume
      ),
    [stocks, query, signal, minimumVolume]
  );

  return (
    <Workspace
      title="Market Screener"
      subtitle={`${filtered.length} instruments match your filters`}
    >
      <div className="grid md:grid-cols-3 gap-3 mb-5">
        <label className="rounded-lg border border-zinc-800 bg-zinc-900 px-4 py-3 flex items-center gap-3">
          <Search size={16} className="text-zinc-500" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search symbol"
            className="bg-transparent outline-none text-sm w-full"
          />
        </label>
        <label className="rounded-lg border border-zinc-800 bg-zinc-900 px-4 py-3 flex items-center gap-3">
          <Filter size={16} className="text-zinc-500" />
          <select
            value={signal}
            onChange={(event) => setSignal(event.target.value)}
            className="bg-zinc-900 outline-none text-sm w-full"
          >
            <option value="ALL">All signals</option>
            <option value="BUY">Buy</option>
            <option value="HOLD">Hold</option>
            <option value="SELL">Sell</option>
          </select>
        </label>
        <label className="rounded-lg border border-zinc-800 bg-zinc-900 px-4 py-2">
          <span className="text-[10px] uppercase text-zinc-500">
            Minimum average volume
          </span>
          <input
            type="number"
            min={0}
            step={1000}
            value={minimumVolume}
            onChange={(event) => setMinimumVolume(Number(event.target.value))}
            className="bg-transparent outline-none text-sm w-full mt-1"
          />
        </label>
      </div>
      <MarketTable stocks={filtered} onOpenStock={onOpenStock} />
    </Workspace>
  );
}

function PortfolioView({
  onOpenStock,
}: {
  stocks: StockSnapshot[];
  onOpenStock: (symbol: string) => void;
}) {
  const [riskProfile, setRiskProfile] =
    useState<PortfolioRiskProfile>("moderate");
  const [companyCount, setCompanyCount] = useState(5);
  const [investmentAmount, setInvestmentAmount] = useState(10000);
  const [result, setResult] = useState<PortfolioOptimizationResponse | null>(
    null
  );
  const [submitting, setSubmitting] = useState(false);
  const [portfolioError, setPortfolioError] = useState<string | null>(null);

  const profiles: Array<{
    value: PortfolioRiskProfile;
    label: string;
    description: string;
  }> = [
    {
      value: "conservative",
      label: "Averse au risque",
      description: "Priorité à la stabilité et aux faibles bêtas.",
    },
    {
      value: "moderate",
      label: "Neutre",
      description: "Équilibre entre rendement CAPM et volatilité.",
    },
    {
      value: "aggressive",
      label: "Preneur de risque",
      description: "Recherche de rendement avec plus de volatilité.",
    },
  ];

  async function submitPortfolio(event: React.FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setPortfolioError(null);
    try {
      setResult(
        await optimizePortfolio({
          risk_profile: riskProfile,
          company_count: companyCount,
          investment_amount: investmentAmount,
        })
      );
    } catch (error) {
      setPortfolioError(
        error instanceof Error
          ? error.message
          : "Impossible de construire le portefeuille."
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Workspace
      title="Portfolio agentique"
      subtitle="CAPM, bêta et portefeuille de variance minimale calculés sur les données BVMT persistées"
    >
      <form
        onSubmit={submitPortfolio}
        className="rounded-xl border border-zinc-800 bg-[#111113] p-5 mb-6"
      >
        <div className="text-xs uppercase tracking-wider text-zinc-500 mb-3">
          1. Votre utilité face au risque
        </div>
        <div className="grid md:grid-cols-3 gap-3 mb-5">
          {profiles.map((profile) => (
            <button
              key={profile.value}
              type="button"
              onClick={() => setRiskProfile(profile.value)}
              className={`rounded-lg border p-4 text-left transition ${
                riskProfile === profile.value
                  ? "border-emerald-500 bg-emerald-500/10"
                  : "border-zinc-800 bg-zinc-950 hover:border-zinc-700"
              }`}
            >
              <div className="font-semibold">{profile.label}</div>
              <div className="text-xs text-zinc-500 mt-1">
                {profile.description}
              </div>
            </button>
          ))}
        </div>

        <div className="grid md:grid-cols-[1fr_1fr_auto] gap-4 items-end">
          <label>
            <span className="block text-xs uppercase text-zinc-500 mb-2">
              Nombre de sociétés
            </span>
            <input
              type="number"
              min={2}
              max={12}
              value={companyCount}
              onChange={(event) =>
                setCompanyCount(
                  Math.min(12, Math.max(2, Number(event.target.value)))
                )
              }
              className="w-full rounded-lg border border-zinc-800 bg-zinc-950 px-4 py-3 outline-none focus:border-emerald-500"
            />
          </label>
          <label>
            <span className="block text-xs uppercase text-zinc-500 mb-2">
              Montant à investir (TND)
            </span>
            <input
              type="number"
              min={100}
              step={100}
              value={investmentAmount}
              onChange={(event) =>
                setInvestmentAmount(Math.max(0, Number(event.target.value)))
              }
              className="w-full rounded-lg border border-zinc-800 bg-zinc-950 px-4 py-3 outline-none focus:border-emerald-500"
            />
          </label>
          <button
            type="submit"
            disabled={submitting || investmentAmount <= 0}
            className="rounded-lg bg-emerald-500 text-zinc-950 font-semibold px-6 py-3 disabled:opacity-50 hover:bg-emerald-400 transition"
          >
            {submitting ? "Calcul en cours…" : "Construire mon PVM"}
          </button>
        </div>
        {portfolioError && (
          <div className="mt-4 text-sm text-red-400">{portfolioError}</div>
        )}
      </form>

      {!result && (
        <div className="rounded-xl border border-dashed border-zinc-800 p-10 text-center text-zinc-500">
          Choisissez votre profil, le nombre de sociétés et votre budget pour
          générer une allocation réelle.
        </div>
      )}

      {result && (
        <>
          <div className="grid sm:grid-cols-2 lg:grid-cols-5 gap-3 mb-6">
            <SummaryCard
              icon={<BriefcaseBusiness size={18} />}
              label="Montant investi"
              value={formatCurrency(result.metrics.invested_amount)}
            />
            <SummaryCard
              icon={<Activity size={18} />}
              label="Rendement CAPM"
              value={formatPercentage(result.metrics.expected_return * 100)}
              positive={result.metrics.expected_return >= 0}
            />
            <SummaryCard
              icon={<BarChart3 size={18} />}
              label="Volatilité"
              value={formatPercentage(result.metrics.volatility * 100)}
            />
            <SummaryCard
              icon={<Activity size={18} />}
              label="Bêta portefeuille"
              value={result.metrics.beta.toFixed(2)}
            />
            <SummaryCard
              icon={<BriefcaseBusiness size={18} />}
              label="Liquidités"
              value={formatCurrency(result.metrics.cash_remaining)}
            />
          </div>

          <div className="grid xl:grid-cols-[1.35fr_.65fr] gap-5 mb-6">
            <div className="rounded-xl border border-zinc-800 overflow-x-auto">
              <table className="w-full text-sm min-w-[760px]">
                <thead className="bg-zinc-900 text-zinc-500 text-xs uppercase">
                  <tr>
                    <th className="text-left px-4 py-3">Société</th>
                    <th className="text-right px-4 py-3">Poids PVM</th>
                    <th className="text-right px-4 py-3">Actions</th>
                    <th className="text-right px-4 py-3">Allocation</th>
                    <th className="text-right px-4 py-3">Bêta</th>
                    <th className="text-right px-4 py-3">CAPM</th>
                  </tr>
                </thead>
                <tbody>
                  {result.assets.map((asset) => (
                    <tr
                      key={asset.symbol}
                      onClick={() => onOpenStock(asset.symbol)}
                      className="border-t border-zinc-800 hover:bg-zinc-900/60 cursor-pointer"
                    >
                      <td className="px-4 py-4 font-semibold">{asset.symbol}</td>
                      <td className="px-4 py-4 text-right font-mono">
                        {(asset.weight * 100).toFixed(2)}%
                      </td>
                      <td className="px-4 py-4 text-right font-mono">
                        {asset.shares}
                      </td>
                      <td className="px-4 py-4 text-right font-mono">
                        {formatCurrency(asset.invested_amount)}
                      </td>
                      <td className="px-4 py-4 text-right font-mono">
                        {asset.beta.toFixed(2)}
                      </td>
                      <td className="px-4 py-4 text-right font-mono">
                        {formatPercentage(asset.capm_return * 100)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="rounded-xl border border-zinc-800 bg-[#111113] p-5">
              <div className="text-xs uppercase tracking-wider text-zinc-500 mb-5">
                Distribution du portefeuille
              </div>
              <div className="space-y-4">
                {result.assets.map((asset) => (
                  <div key={asset.symbol}>
                    <div className="flex justify-between text-sm mb-1.5">
                      <span className="font-semibold">{asset.symbol}</span>
                      <span className="font-mono">
                        {(asset.weight * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2 rounded-full bg-zinc-800 overflow-hidden">
                      <div
                        className="h-full rounded-full bg-emerald-500"
                        style={{ width: `${asset.weight * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="grid xl:grid-cols-[.9fr_1.1fr] gap-5 mb-6">
            <div className="rounded-xl border border-zinc-800 bg-[#111113] p-5">
              <div className="text-xs uppercase tracking-wider text-zinc-500 mb-4">
                Comment les poids sont calculés
              </div>
              <div className="space-y-4 text-sm text-zinc-300">
                <FormulaBlock
                  title="Bêta"
                  formula="βᵢ = Cov(Rᵢ, Rₘ) / Var(Rₘ)"
                  detail={`${result.assets[0].symbol}: ${result.assets[0].covariance_with_market.toExponential(3)} / ${result.methodology.market_variance.toExponential(3)} = ${result.assets[0].beta.toFixed(3)}`}
                />
                <FormulaBlock
                  title="CAPM / MEDAF"
                  formula="E(Rᵢ) = Rf + βᵢ × [E(Rₘ) − Rf]"
                  detail={`Portefeuille: ${(result.metrics.risk_free_rate * 100).toFixed(1)}% + ${result.metrics.beta.toFixed(3)} × ${(result.methodology.market_risk_premium * 100).toFixed(2)}% = ${(result.metrics.expected_return * 100).toFixed(2)}%`}
                />
                <FormulaBlock
                  title="Distribution PVM"
                  formula="min wᵀΣw  sous  Σwᵢ = 1"
                  detail={`Variance minimale sous contraintes: ${(result.methodology.minimum_weight * 100).toFixed(0)}% ≤ wᵢ ≤ ${(result.methodology.maximum_weight * 100).toFixed(0)}%. Covariance annualisée sur ${result.methodology.observations} séances.`}
                />
                <p className="text-xs leading-5 text-zinc-500">
                  L’optimiseur teste les pondérations autorisées et retient celles
                  qui minimisent simultanément les variances individuelles et les
                  corrélations entre sociétés. Les montants sont ensuite
                  `poids × capital`, puis convertis en actions entières.
                </p>
              </div>
            </div>

            <div className="rounded-xl border border-zinc-800 bg-[#111113] p-5">
              <div className="flex items-start justify-between gap-4 mb-4">
                <div>
                  <div className="text-xs uppercase tracking-wider text-zinc-500">
                    Frontière efficiente PVM
                  </div>
                  <p className="text-xs text-zinc-500 mt-1">
                    Rendement CAPM maximal pour chaque niveau de risque admissible
                  </p>
                </div>
                <Badge variant="success">PVM sélectionné</Badge>
              </div>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={result.efficient_frontier.map((point) => ({
                      risk: point.volatility * 100,
                      return: point.expected_return * 100,
                    }))}
                    margin={{ top: 15, right: 20, bottom: 15, left: 5 }}
                  >
                    <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
                    <XAxis
                      type="number"
                      dataKey="risk"
                      domain={["dataMin", "dataMax"]}
                      tickFormatter={(value) => `${Number(value).toFixed(1)}%`}
                      stroke="#71717a"
                      label={{
                        value: "Volatilité annualisée",
                        position: "insideBottom",
                        offset: -8,
                        fill: "#71717a",
                      }}
                    />
                    <YAxis
                      type="number"
                      domain={["dataMin", "dataMax"]}
                      tickFormatter={(value) => `${Number(value).toFixed(1)}%`}
                      stroke="#71717a"
                      width={58}
                    />
                    <Tooltip
                      formatter={(value: number) => `${value.toFixed(2)}%`}
                      labelFormatter={(value) =>
                        `Risque: ${Number(value).toFixed(2)}%`
                      }
                      contentStyle={{
                        background: "#18181b",
                        border: "1px solid #3f3f46",
                        borderRadius: 8,
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="return"
                      name="Rendement CAPM"
                      stroke="#10b981"
                      strokeWidth={3}
                      dot={{ fill: "#10b981", r: 3 }}
                    />
                    <ReferenceDot
                      x={result.metrics.volatility * 100}
                      y={result.metrics.expected_return * 100}
                      r={7}
                      fill="#f59e0b"
                      stroke="#fef3c7"
                      label={{ value: "PVM", position: "top", fill: "#f59e0b" }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="rounded-xl border border-emerald-900/70 bg-emerald-950/20 p-5">
            <div className="flex items-center justify-between gap-3 mb-3">
              <div className="text-xs uppercase tracking-wider text-emerald-400">
                Explication multi-agent
              </div>
            </div>
            <p className="text-sm leading-7 text-zinc-300 whitespace-pre-line">
              {result.explanation}
            </p>
            {result.warnings.map((warning) => (
              <div key={warning} className="text-xs text-amber-400 mt-3">
                {warning}
              </div>
            ))}
          </div>
        </>
      )}
    </Workspace>
  );
}

function FormulaBlock({
  title,
  formula,
  detail,
}: {
  title: string;
  formula: string;
  detail: string;
}) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-4">
      <div className="text-xs font-semibold uppercase text-emerald-400 mb-2">
        {title}
      </div>
      <div className="font-mono text-base text-white mb-2">{formula}</div>
      <div className="font-mono text-xs leading-5 text-zinc-500">{detail}</div>
    </div>
  );
}

function MarketTable({
  stocks,
  onOpenStock,
}: {
  stocks: StockSnapshot[];
  onOpenStock: (symbol: string) => void;
}) {
  return (
    <div className="rounded-xl border border-zinc-800 overflow-x-auto">
      <table className="w-full text-sm min-w-[760px]">
        <thead className="bg-zinc-900 text-zinc-500 text-xs uppercase">
          <tr>
            <th className="text-left px-4 py-3">Symbol</th>
            <th className="text-right px-4 py-3">Close</th>
            <th className="text-right px-4 py-3">Change</th>
            <th className="text-right px-4 py-3">Volume</th>
            <th className="text-right px-4 py-3">Sentiment</th>
            <th className="text-right px-4 py-3">Signal</th>
            <th className="text-right px-4 py-3">Alerts</th>
          </tr>
        </thead>
        <tbody>
          {stocks.map((stock) => (
            <tr
              key={stock.symbol}
              onClick={() => onOpenStock(stock.symbol)}
              className="border-t border-zinc-800 hover:bg-zinc-900/60 cursor-pointer"
            >
              <td className="px-4 py-3 font-semibold text-white">{stock.symbol}</td>
              <td className="px-4 py-3 text-right font-mono">
                {formatCurrency(stock.price)}
              </td>
              <td
                className={`px-4 py-3 text-right font-mono ${
                  stock.change >= 0 ? "text-emerald-400" : "text-red-400"
                }`}
              >
                {stock.changePercent >= 0 ? "+" : ""}
                {stock.changePercent.toFixed(2)}%
              </td>
              <td className="px-4 py-3 text-right font-mono text-zinc-300">
                {formatNumber(stock.volume)}
              </td>
              <td className="px-4 py-3 text-right font-mono">
                {stock.sentimentScore.toFixed(2)}
              </td>
              <td className="px-4 py-3 text-right">
                <SignalBadge signal={stock.recommendation} />
              </td>
              <td className="px-4 py-3 text-right">{stock.anomalyCount || 0}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Workspace({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <main className="flex-1 p-4 md:p-8 overflow-y-auto">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-white">{title}</h1>
        <p className="text-zinc-500 mt-1 mb-6">{subtitle}</p>
        {children}
      </div>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="text-xs text-zinc-500 uppercase font-semibold tracking-wide">
        {label}
      </span>
      <div className="font-mono text-zinc-100">{value}</div>
    </div>
  );
}

function SignalBadge({ signal }: { signal?: "BUY" | "SELL" | "HOLD" }) {
  if (!signal) return <span className="text-zinc-600">—</span>;
  return (
    <Badge
      variant={
        signal === "BUY" ? "success" : signal === "SELL" ? "danger" : "neutral"
      }
    >
      {signal}
    </Badge>
  );
}

function SummaryCard({
  icon,
  label,
  value,
  positive,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  positive?: boolean;
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
      <div className="flex items-center gap-2 text-zinc-500 text-xs uppercase">
        {icon}
        {label}
      </div>
      <div
        className={`text-2xl font-mono mt-3 ${
          positive === undefined
            ? "text-white"
            : positive
              ? "text-emerald-400"
              : "text-red-400"
        }`}
      >
        {value}
      </div>
    </div>
  );
}
