/* eslint-disable no-unused-vars */
import { useState, useMemo, useEffect } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend, LineChart, Line, XAxis, YAxis, CartesianGrid } from 'recharts';
import { Plus, Lightbulb, AlertTriangle, TrendingUp, TrendingDown, Target, Sparkles, Wallet, DollarSign, BarChart3 } from 'lucide-react';
import { chartConfig, chartColors } from '../utils/chartConfig';
import { useStockData } from '../hooks/useStockData';
import {
  usePortfolioSnapshot,
  usePortfolioPerformance,
  usePerformanceExplanation,
  useCreatePortfolio,
  useExecuteTrade,
  useUpdatePrices,
  getStoredPortfolioId,
} from '../hooks/usePortfolio';

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4'];

const Portfolio = () => {
  const { stocks } = useStockData();
  const portfolioId = getStoredPortfolioId();

  const { snapshot, isLoading: isSnapLoading, refetch: refetchSnap } = usePortfolioSnapshot(portfolioId);
  const { performance, isLoading: isPerfLoading } = usePortfolioPerformance(portfolioId);
  const { explanation } = usePerformanceExplanation(portfolioId);

  const createPortfolioMut = useCreatePortfolio();
  const tradeMutation = useExecuteTrade(portfolioId);
  const updatePrices = useUpdatePrices(portfolioId);

  const [createProfile, setCreateProfile] = useState('moderate');
  const [createCapital, setCreateCapital] = useState(10000);

  const [showTrade, setShowTrade] = useState(false);
  const [tradeSymbol, setTradeSymbol] = useState('');
  const [tradeAction, setTradeAction] = useState('buy');
  const [tradeQty, setTradeQty] = useState(10);
  const [tradePrice, setTradePrice] = useState('');

  // Auto-update portfolio prices when stocks data arrives
  useEffect(() => {
    if (portfolioId && stocks?.length > 0 && snapshot?.positions) {
      const prices = {};
      const positionSymbols = Object.keys(snapshot.positions);
      stocks.forEach(s => {
        if (positionSymbols.includes(s.ticker)) {
          prices[s.ticker] = parseFloat(s.dernier) || 0;
        }
      });
      if (Object.keys(prices).length > 0) {
        updatePrices.mutate(prices);
      }
    }
  }, [stocks]); // eslint-disable-line react-hooks/exhaustive-deps

  // Derived portfolio data
  const positions = useMemo(() => {
    if (!snapshot?.positions) return [];
    return Object.entries(snapshot.positions).map(([symbol, pos]) => ({
      symbol,
      ...pos,
      currentPrice: pos.current_price || pos.avg_price,
      avgPrice: pos.avg_price,
      quantity: pos.quantity,
      value: (pos.current_price || pos.avg_price) * pos.quantity,
      invested: pos.avg_price * pos.quantity,
      pl: ((pos.current_price || pos.avg_price) - pos.avg_price) * pos.quantity,
      plPercent: pos.avg_price > 0 ? (((pos.current_price || pos.avg_price) - pos.avg_price) / pos.avg_price * 100) : 0,
    }));
  }, [snapshot]);

  const portfolioMetrics = useMemo(() => {
    if (!snapshot) return null;
    return {
      totalValue: snapshot.total_value || 0,
      cash: snapshot.cash || 0,
      totalInvested: positions.reduce((s, p) => s + p.invested, 0),
      currentValue: positions.reduce((s, p) => s + p.value, 0),
      totalPL: positions.reduce((s, p) => s + p.pl, 0),
      positionCount: positions.length,
    };
  }, [snapshot, positions]);

  // Sector allocation
  const sectorData = useMemo(() => {
    if (positions.length === 0) return [];
    const sectors = {};
    positions.forEach(pos => {
      const matchedStock = stocks?.find(s => s.ticker === pos.symbol);
      const sector = matchedStock?.groupe || matchedStock?.valGroup || 'Autre';
      if (!sectors[sector]) sectors[sector] = 0;
      sectors[sector] += pos.value;
    });
    const total = Object.values(sectors).reduce((s, v) => s + v, 0);
    return Object.entries(sectors).map(([name, value]) => ({
      name,
      value: parseFloat(value.toFixed(2)),
      percentage: total > 0 ? ((value / total) * 100).toFixed(1) : '0.0',
    }));
  }, [positions, stocks]);

  const perfMetrics = useMemo(() => {
    if (!performance) return null;
    return {
      roi: performance.roi != null ? (performance.roi * 100).toFixed(2) : '–',
      sharpe: performance.sharpe_ratio != null ? performance.sharpe_ratio.toFixed(2) : '–',
      maxDrawdown: performance.max_drawdown != null ? (performance.max_drawdown * 100).toFixed(2) : '–',
      volatility: performance.volatility != null ? (performance.volatility * 100).toFixed(2) : '–',
      winRate: performance.win_rate != null ? (performance.win_rate * 100).toFixed(0) : '–',
      totalTrades: performance.total_trades || 0,
    };
  }, [performance]);

  // Create portfolio handler
  const handleCreate = () => {
    createPortfolioMut.mutate(
      { riskProfile: createProfile, initialCapital: createCapital },
      { onSuccess: () => refetchSnap() }
    );
  };

  // Trade handler
  const handleTrade = () => {
    if (!tradeSymbol || !tradePrice) return;
    tradeMutation.mutate(
      { symbol: tradeSymbol.toUpperCase(), action: tradeAction, quantity: tradeQty, price: parseFloat(tradePrice) },
      { onSuccess: () => { setShowTrade(false); setTradeSymbol(''); setTradePrice(''); } }
    );
  };

  // If no portfolio exists and not loading
  if (!isSnapLoading && !snapshot && !portfolioId) {
    return (
      <div className="p-4 md:p-6 lg:p-8">
        <div className="max-w-[1600px] mx-auto">
          <div className="finance-card p-12 rounded-lg text-center">
            <Wallet className="w-16 h-16 text-primary-500 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-finance-text-primary mb-2">Créer un Portefeuille</h2>
            <p className="text-finance-text-secondary mb-6">
              Commencez par créer un portefeuille virtuel pour suivre vos investissements BVMT.
            </p>
            <div className="max-w-md mx-auto space-y-4">
              <div>
                <label className="block text-sm font-medium text-finance-text-primary mb-1">Profil de Risque</label>
                <select
                  value={createProfile}
                  onChange={e => setCreateProfile(e.target.value)}
                  className="w-full px-4 py-3 bg-finance-card border border-finance-border rounded-lg text-finance-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500"
                >
                  <option value="conservative">Conservateur</option>
                  <option value="moderate">Modéré</option>
                  <option value="aggressive">Agressif</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-finance-text-primary mb-1">Capital Initial (TND)</label>
                <input
                  type="number"
                  value={createCapital}
                  onChange={e => setCreateCapital(Number(e.target.value))}
                  min={1000}
                  className="w-full px-4 py-3 bg-finance-card border border-finance-border rounded-lg text-finance-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
              </div>
              <button
                onClick={handleCreate}
                disabled={createPortfolioMut.isPending}
                className="w-full px-6 py-3 bg-primary-500 text-white rounded-lg font-medium hover:bg-primary-600 transition-colors disabled:opacity-50"
              >
                {createPortfolioMut.isPending ? 'Création...' : 'Créer le Portefeuille'}
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (isSnapLoading) {
    return (
      <div className="p-4 md:p-6 lg:p-8">
        <div className="max-w-[1600px] mx-auto">
          <div className="animate-pulse space-y-6">
            <div className="h-12 bg-finance-bg rounded w-1/3"></div>
            <div className="grid grid-cols-4 gap-4">
              {[1, 2, 3, 4].map(i => <div key={i} className="h-24 bg-finance-bg rounded"></div>)}
            </div>
            <div className="h-64 bg-finance-bg rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 md:p-6 lg:p-8">
      <div className="max-w-[1600px] mx-auto">
        {/* Header */}
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold text-finance-text-primary mb-2">
              Mon Portefeuille
            </h1>
            <p className="text-finance-text-secondary">
              Gérez et suivez vos investissements en temps réel
            </p>
          </div>
          <button
            onClick={() => setShowTrade(!showTrade)}
            className="px-6 py-3 bg-primary-500 text-white rounded-lg font-medium hover:bg-primary-600 transition-colors flex items-center gap-2"
          >
            <Plus className="w-5 h-5" />
            Exécuter un Trade
          </button>
        </div>

        {/* Trade form */}
        {showTrade && (
          <div className="finance-card p-6 rounded-lg mb-6">
            <h3 className="text-lg font-semibold text-finance-text-primary mb-4">Nouveau Trade</h3>
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4 items-end">
              <div>
                <label className="block text-sm text-finance-text-secondary mb-1">Symbole</label>
                <select
                  value={tradeSymbol}
                  onChange={e => {
                    setTradeSymbol(e.target.value);
                    const s = stocks?.find(st => st.ticker === e.target.value);
                    if (s) setTradePrice(s.dernier);
                  }}
                  className="w-full px-3 py-2 bg-finance-bg border border-finance-border rounded text-finance-text-primary"
                >
                  <option value="">Sélectionner</option>
                  {stocks?.map(s => (
                    <option key={s.ticker} value={s.ticker}>{s.ticker}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm text-finance-text-secondary mb-1">Action</label>
                <select
                  value={tradeAction}
                  onChange={e => setTradeAction(e.target.value)}
                  className="w-full px-3 py-2 bg-finance-bg border border-finance-border rounded text-finance-text-primary"
                >
                  <option value="buy">Acheter</option>
                  <option value="sell">Vendre</option>
                </select>
              </div>
              <div>
                <label className="block text-sm text-finance-text-secondary mb-1">Quantité</label>
                <input
                  type="number" min={1} value={tradeQty}
                  onChange={e => setTradeQty(Number(e.target.value))}
                  className="w-full px-3 py-2 bg-finance-bg border border-finance-border rounded text-finance-text-primary"
                />
              </div>
              <div>
                <label className="block text-sm text-finance-text-secondary mb-1">Prix (TND)</label>
                <input
                  type="number" step="0.001" value={tradePrice}
                  onChange={e => setTradePrice(e.target.value)}
                  className="w-full px-3 py-2 bg-finance-bg border border-finance-border rounded text-finance-text-primary"
                />
              </div>
              <button
                onClick={handleTrade}
                disabled={tradeMutation.isPending || !tradeSymbol || !tradePrice}
                className="px-4 py-2 bg-success-500 text-white rounded font-medium hover:bg-success-600 transition-colors disabled:opacity-50"
              >
                {tradeMutation.isPending ? '...' : 'Valider'}
              </button>
            </div>
            {tradeMutation.isError && (
              <p className="mt-2 text-sm text-danger-500">
                Erreur: {tradeMutation.error?.message || 'Trade échoué'}
              </p>
            )}
          </div>
        )}

        {/* Key Metrics */}
        {portfolioMetrics && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <div className="finance-card p-6 rounded-lg">
              <p className="text-sm text-finance-text-secondary mb-2">Valeur Totale</p>
              <p className="text-2xl font-bold text-finance-text-primary">{portfolioMetrics.totalValue.toFixed(2)} TND</p>
              <p className="text-sm text-finance-text-secondary mt-1">{portfolioMetrics.positionCount} position(s)</p>
            </div>
            <div className="finance-card p-6 rounded-lg">
              <p className="text-sm text-finance-text-secondary mb-2">P&L Total</p>
              <p className={`text-2xl font-bold ${portfolioMetrics.totalPL >= 0 ? 'text-success-500' : 'text-danger-500'}`}>
                {portfolioMetrics.totalPL >= 0 ? '+' : ''}{portfolioMetrics.totalPL.toFixed(2)} TND
              </p>
              {portfolioMetrics.totalInvested > 0 && (
                <p className={`text-sm font-semibold mt-1 ${portfolioMetrics.totalPL >= 0 ? 'text-success-500' : 'text-danger-500'}`}>
                  {portfolioMetrics.totalPL >= 0 ? '+' : ''}{((portfolioMetrics.totalPL / portfolioMetrics.totalInvested) * 100).toFixed(2)}%
                </p>
              )}
            </div>
            <div className="finance-card p-6 rounded-lg">
              <p className="text-sm text-finance-text-secondary mb-2">ROI Global</p>
              <p className={`text-2xl font-bold ${perfMetrics && parseFloat(perfMetrics.roi) >= 0 ? 'text-success-500' : 'text-danger-500'}`}>
                {perfMetrics ? `${parseFloat(perfMetrics.roi) >= 0 ? '+' : ''}${perfMetrics.roi}%` : '–'}
              </p>
              <p className="text-sm text-finance-text-secondary mt-1">Depuis le début</p>
            </div>
            <div className="finance-card p-6 rounded-lg">
              <p className="text-sm text-finance-text-secondary mb-2">Liquidités</p>
              <p className="text-2xl font-bold text-finance-text-primary">{portfolioMetrics.cash.toFixed(2)} TND</p>
              <p className="text-sm text-finance-text-secondary mt-1">Disponible</p>
            </div>
          </div>
        )}

        {/* Performance metrics row */}
        {perfMetrics && (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-6">
            <div className="finance-card p-4 rounded-lg text-center">
              <p className="text-xs text-finance-text-secondary mb-1">Sharpe Ratio</p>
              <p className="text-xl font-bold text-finance-text-primary">{perfMetrics.sharpe}</p>
            </div>
            <div className="finance-card p-4 rounded-lg text-center">
              <p className="text-xs text-finance-text-secondary mb-1">Max Drawdown</p>
              <p className="text-xl font-bold text-danger-500">{perfMetrics.maxDrawdown}%</p>
            </div>
            <div className="finance-card p-4 rounded-lg text-center">
              <p className="text-xs text-finance-text-secondary mb-1">Volatilité</p>
              <p className="text-xl font-bold text-warning-500">{perfMetrics.volatility}%</p>
            </div>
            <div className="finance-card p-4 rounded-lg text-center">
              <p className="text-xs text-finance-text-secondary mb-1">Win Rate</p>
              <p className="text-xl font-bold text-success-500">{perfMetrics.winRate}%</p>
            </div>
            <div className="finance-card p-4 rounded-lg text-center">
              <p className="text-xs text-finance-text-secondary mb-1">Total Trades</p>
              <p className="text-xl font-bold text-finance-text-primary">{perfMetrics.totalTrades}</p>
            </div>
          </div>
        )}

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Sector Distribution */}
          {sectorData.length > 0 && (
            <div className="finance-card p-6 rounded-lg">
              <h3 className="text-lg font-semibold text-finance-text-primary mb-4">
                Répartition par Secteur
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={sectorData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percentage }) => `${name} (${percentage}%)`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {sectorData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    {...chartConfig.tooltip}
                    formatter={(value) => `${value} TND`}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* AI Explanation */}
          {explanation && (
            <div className="finance-card p-6 rounded-lg">
              <h3 className="text-lg font-semibold text-finance-text-primary mb-4 flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-primary-500" />
                Analyse IA du Portefeuille
              </h3>
              <p className="text-finance-text-primary leading-relaxed whitespace-pre-line">{explanation}</p>
            </div>
          )}
        </div>

        {/* Positions Table */}
        {positions.length > 0 && (
          <div className="finance-card p-6 rounded-lg mb-6">
            <h3 className="text-lg font-semibold text-finance-text-primary mb-4">
              Positions Actuelles
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-finance-border">
                    <th className="text-left py-3 px-2 text-sm font-semibold text-finance-text-secondary">Valeur</th>
                    <th className="text-right py-3 px-2 text-sm font-semibold text-finance-text-secondary">Quantité</th>
                    <th className="text-right py-3 px-2 text-sm font-semibold text-finance-text-secondary">Prix Moyen</th>
                    <th className="text-right py-3 px-2 text-sm font-semibold text-finance-text-secondary">Prix Actuel</th>
                    <th className="text-right py-3 px-2 text-sm font-semibold text-finance-text-secondary">Valeur</th>
                    <th className="text-right py-3 px-2 text-sm font-semibold text-finance-text-secondary">P&L</th>
                    <th className="text-right py-3 px-2 text-sm font-semibold text-finance-text-secondary">P&L %</th>
                  </tr>
                </thead>
                <tbody>
                  {positions.map((pos) => (
                    <tr key={pos.symbol} className="border-b border-finance-border hover:bg-finance-bg transition-colors">
                      <td className="py-4 px-2">
                        <p className="font-semibold text-finance-text-primary">{pos.symbol}</p>
                      </td>
                      <td className="text-right py-4 px-2 text-finance-text-primary">{pos.quantity}</td>
                      <td className="text-right py-4 px-2 text-finance-text-primary">{pos.avgPrice.toFixed(3)}</td>
                      <td className="text-right py-4 px-2 text-finance-text-primary">{pos.currentPrice.toFixed(3)}</td>
                      <td className="text-right py-4 px-2 font-semibold text-finance-text-primary">{pos.value.toFixed(2)}</td>
                      <td className={`text-right py-4 px-2 font-semibold ${pos.pl >= 0 ? 'text-success-500' : 'text-danger-500'}`}>
                        {pos.pl >= 0 ? '+' : ''}{pos.pl.toFixed(2)}
                      </td>
                      <td className={`text-right py-4 px-2 font-semibold ${pos.plPercent >= 0 ? 'text-success-500' : 'text-danger-500'}`}>
                        {pos.plPercent >= 0 ? '+' : ''}{pos.plPercent.toFixed(2)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {positions.length === 0 && (
          <div className="finance-card p-12 rounded-lg text-center mb-6">
            <BarChart3 className="w-12 h-12 mx-auto mb-2 text-finance-text-secondary" />
            <p className="text-finance-text-secondary">
              Aucune position. Utilisez le bouton "Exécuter un Trade" pour acheter votre premier titre.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Portfolio;
