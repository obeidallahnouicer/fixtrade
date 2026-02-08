import { useState } from 'react';
import { useStockData } from '../hooks/useStockData';
import { useCalculateTunindex } from '../hooks/useCalculateTunindex';
import { useRecentAnomalies } from '../hooks/useAnomalies';
import IndexCard from '../components/IndexCard';
import StockCard from '../components/StockCard';
import GainersLosers from '../components/GainersLosers';
import StockDetail from '../components/StockDetail';
import TunindexChart from '../components/TunindexChart';
import StockFilter from '../components/StockFilter';
import Skeleton from '../components/Skeleton';
import { TrendingUp, TrendingDown, Target, AlertTriangle, Flame, Snowflake } from 'lucide-react';

const MarketOverview = () => {
  const { stocks, isLoading, isError, error, dataUpdatedAt } = useStockData();
  const tunindexData = useCalculateTunindex(stocks);
  const { data: backendAnomalies } = useRecentAnomalies();
  const [selectedStock, setSelectedStock] = useState(null);
  const [isDetailOpen, setIsDetailOpen] = useState(false);
  const [filteredStocks, setFilteredStocks] = useState([]);
  const [comparisonStock, setComparisonStock] = useState(null);

  const handleStockClick = (stock) => {
    setSelectedStock(stock);
    setIsDetailOpen(true);
  };

  const handleCloseDetail = () => {
    setIsDetailOpen(false);
    setTimeout(() => setSelectedStock(null), 300);
  };

  // Loading state
  if (isLoading) {
    return <Skeleton.Dashboard />;
  }

  // Error state
  if (isError) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <div className="finance-card rounded-lg p-8 max-w-md w-full text-center">
          <svg className="w-16 h-16 text-danger-500 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <h2 className="text-xl font-semibold text-finance-text-primary mb-2">
            Unable to Load Data
          </h2>
          <p className="text-finance-text-secondary text-sm mb-4">
            {error?.message || 'An error occurred while fetching stock data'}
          </p>
          <button
            onClick={() => window.location.reload()}
            className="px-6 py-2.5 bg-primary-500 text-white rounded font-medium hover:bg-primary-600 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const displayStocks = filteredStocks.length > 0 ? filteredStocks : stocks;

  // Calculate market sentiment
  const calculateMarketSentiment = () => {
    const posStocks = stocks.filter(s => s.variation > 0).length;
    const negStocks = stocks.filter(s => s.variation < 0).length;
    
    if (posStocks > negStocks * 1.5) return { label: 'Très Positif', color: 'text-success-500', icon: TrendingUp };
    if (posStocks > negStocks) return { label: 'Positif', color: 'text-success-400', icon: TrendingUp };
    if (negStocks > posStocks * 1.5) return { label: 'Très Négatif', color: 'text-danger-500', icon: TrendingDown };
    if (negStocks > posStocks) return { label: 'Négatif', color: 'text-danger-400', icon: TrendingDown };
    return { label: 'Neutre', color: 'text-gray-400', icon: Target };
  };

  const sentiment = calculateMarketSentiment();

  return (
    <div className="p-4 md:p-6 lg:p-8">
      <div className="max-w-[1600px] mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl md:text-3xl font-bold text-finance-text-primary mb-2">
            Vue d'Ensemble du Marché
          </h1>
          <p className="text-finance-text-secondary">
            Bourse des Valeurs Mobilières de Tunis
          </p>
        </div>

        {/* TUNINDEX Card */}
        <IndexCard
          value={tunindexData.indexValue}
          percentageChange={tunindexData.percentageChange}
          change={tunindexData.change}
          lastUpdated={dataUpdatedAt}
        />

        {/* Market Sentiment Card */}
        <div className="mb-6">
          <div className="finance-card p-6 rounded-lg">
            <h2 className="text-lg font-semibold text-finance-text-primary mb-4 flex items-center gap-2">
              <Target className="w-5 h-5" />
              Sentiment Global du Marché
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center">
                <div className={`flex justify-center mb-2 ${sentiment.color}`}>
                  <sentiment.icon className="w-12 h-12" />
                </div>
                <p className={`text-xl font-semibold mt-2 ${sentiment.color}`}>
                  {sentiment.label}
                </p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-success-500">
                  {stocks.filter(s => s.variation > 0).length}
                </p>
                <p className="text-sm text-finance-text-secondary mt-1">En hausse</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-danger-500">
                  {stocks.filter(s => s.variation < 0).length}
                </p>
                <p className="text-sm text-finance-text-secondary mt-1">En baisse</p>
              </div>
            </div>
          </div>
        </div>

        {/* Recent Alerts Section */}
        <div className="mb-6">
          <div className="finance-card p-6 rounded-lg">
            <h2 className="text-lg font-semibold text-finance-text-primary mb-4 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5" />
              Alertes Récentes
            </h2>
            <div className="space-y-3">
              {/* Backend AI anomalies */}
              {(backendAnomalies || []).slice(0, 3).map((a, idx) => (
                <div key={`ai-${idx}`} className="flex items-center justify-between p-3 bg-finance-bg rounded border-l-4 border-primary-500">
                  <div className="flex items-center gap-3">
                    <div className="text-primary-400">
                      <AlertTriangle className="w-5 h-5" />
                    </div>
                    <div>
                      <p className="font-semibold text-finance-text-primary flex items-center gap-2">
                        {a.symbol || a.ticker}
                        <span className="text-[10px] bg-primary-500/20 text-primary-400 px-1.5 py-0.5 rounded font-medium">IA</span>
                      </p>
                      <p className="text-sm text-finance-text-secondary">
                        {a.anomaly_type || a.type || 'Anomalie détectée par IA'}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="font-bold text-primary-400">
                      Score: {typeof a.severity_score === 'number' ? a.severity_score.toFixed(1) : (a.severity || 'N/A')}
                    </p>
                    <p className="text-xs text-finance-text-secondary">
                      {a.detected_at ? new Date(a.detected_at).toLocaleTimeString('fr-FR') : 'Récent'}
                    </p>
                  </div>
                </div>
              ))}
              {/* Live variation alerts */}
              {stocks
                .filter(s => Math.abs(s.variation) > 3)
                .sort((a, b) => Math.abs(b.variation) - Math.abs(a.variation))
                .slice(0, 5)
                .map((stock, idx) => (
                  <div key={`live-${idx}`} className="flex items-center justify-between p-3 bg-finance-bg rounded">
                    <div className="flex items-center gap-3">
                      <div className={stock.variation > 0 ? 'text-danger-500' : 'text-primary-400'}>
                        {stock.variation > 0 ? <Flame className="w-5 h-5" /> : <Snowflake className="w-5 h-5" />}
                      </div>
                      <div>
                        <p className="font-semibold text-finance-text-primary">{stock.ticker}</p>
                        <p className="text-sm text-finance-text-secondary">
                          {stock.variation > 0 ? 'Forte hausse détectée' : 'Forte baisse détectée'}
                        </p>
                      </div>
                    </div>
                    <div className={`text-right ${stock.variation >= 0 ? 'text-success-500' : 'text-danger-500'}`}>
                      <p className="font-bold">{stock.variation >= 0 ? '+' : ''}{stock.variation.toFixed(2)}%</p>
                      <p className="text-xs text-finance-text-secondary">Aujourd'hui</p>
                    </div>
                  </div>
                ))}
              {(!backendAnomalies || backendAnomalies.length === 0) && stocks.filter(s => Math.abs(s.variation) > 3).length === 0 && (
                <p className="text-center text-finance-text-secondary py-4">Aucune alerte récente</p>
              )}
            </div>
          </div>
        </div>

        {/* TUNINDEX Chart */}
        <TunindexChart 
          indexData={tunindexData}
          comparisonStock={comparisonStock}
        />

        {/* Stock Filter */}
        <StockFilter
          stocks={stocks}
          tunindexPerformance={tunindexData.percentageChange}
          onFilterChange={setFilteredStocks}
          onStockSelect={setComparisonStock}
        />

        {/* Gainers and Losers */}
        <GainersLosers stocks={stocks} />

        {/* All Stocks Grid */}
        <div className="mb-6">
          <h2 className="text-xl md:text-2xl font-semibold text-finance-text-primary mb-4">
            {filteredStocks.length > 0 ? `Valeurs Filtrées (${displayStocks.length})` : 'Toutes les Valeurs'}
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {displayStocks.map((stock) => (
              <StockCard
                key={stock.id || stock.ticker}
                stock={stock}
                onClick={handleStockClick}
              />
            ))}
          </div>
        </div>

        {/* Stock Detail Modal */}
        <StockDetail
          stock={selectedStock}
          isOpen={isDetailOpen}
          onClose={handleCloseDetail}
        />
      </div>
    </div>
  );
};

export default MarketOverview;
