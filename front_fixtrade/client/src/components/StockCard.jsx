import { memo } from 'react';
import { formatCurrency, formatPercentage, getPriceColorClass, formatLargeNumber } from '../utils/formatters';
import { useStockPrediction } from '../hooks/useStockPredictions';

const StockCard = memo(({ stock, onClick }) => {
  const colorClass = getPriceColorClass(stock.percentChange);
  
  // Better data extraction with fallbacks
  const stockName = stock.stockName || stock.referentiel?.stockName || 'Unknown Stock';
  const ticker = stock.ticker || stock.referentiel?.ticker || 'N/A';
  const lastPrice = parseFloat(stock.last) || 0;
  const change = parseFloat(stock.change) || 0;
  const percentChange = stock.percentChange || 0;
  const volume = parseFloat(stock.volume) || 0;
  const marketCap = parseFloat(stock.caps) || 0;
  const valGroup = stock.valGroup || 'N/A';

  // Fetch prediction for next day
  const { prediction, isLoading: isPredictionLoading, isError: isPredictionError } = useStockPrediction(ticker, 1);

  return (
    <div
      onClick={() => onClick && onClick(stock)}
      className="finance-card-hover rounded-lg p-4 cursor-pointer min-h-[160px] flex flex-col justify-between transition-colors duration-200"
    >
      <div>
        <div className="flex items-start justify-between mb-2">
          <div className="flex-1 min-w-0">
            <h3 className="text-sm md:text-base font-semibold text-finance-text-primary truncate" title={stockName}>
              {stockName}
            </h3>
            <p className="text-xs text-finance-text-secondary">
              {ticker}
            </p>
          </div>
          <div className={`text-xs font-medium px-2 py-1 rounded ${
            valGroup === "11" ? 'bg-primary-500/20 text-primary-400' : 'bg-finance-border text-finance-text-secondary'
          }`}>
            {valGroup === "11" ? 'Primary' : valGroup === 'N/A' ? 'N/A' : `G${valGroup}`}
          </div>
        </div>

        <div className="mb-2">
          <div className="flex items-baseline gap-2 mb-1">
            <div className="text-2xl md:text-3xl font-bold text-finance-text-primary tab-num">
              {lastPrice > 0 ? formatCurrency(lastPrice) : 'N/A'}
            </div>
            <span className="text-xs text-finance-text-secondary font-medium">Current</span>
          </div>
          
          {/* Predicted Price */}
          {!isPredictionError && prediction && (
            <div className="flex items-baseline gap-2 mb-1">
              <div className="text-lg md:text-xl font-semibold text-primary-400 tab-num">
                {formatCurrency(parseFloat(prediction.predicted_close))}
              </div>
              <span className="text-xs text-primary-400/70 font-medium">Predicted</span>
              <svg className="w-3 h-3 text-primary-400/70" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
            </div>
          )}

          {/* Prediction Loading State */}
          {isPredictionLoading && (
            <div className="flex items-baseline gap-2 mb-1">
              <div className="h-5 md:h-6 w-20 bg-finance-border animate-pulse rounded"></div>
              <span className="text-xs text-finance-text-secondary font-medium">Loading...</span>
            </div>
          )}

          <div className={`flex items-center gap-1 text-sm md:text-base font-semibold ${colorClass} mt-1`}>
            {percentChange > 0 && <span>↑</span>}
            {percentChange < 0 && <span>↓</span>}
            {percentChange === 0 && <span>→</span>}
            <span className="tab-num">{formatPercentage(percentChange)}</span>
            <span className="text-xs tab-num">({change > 0 ? '+' : ''}{change.toFixed(2)})</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs text-finance-text-secondary pt-2 border-t border-finance-border">
        <div>
          <div className="font-medium">Volume</div>
          <div className="font-semibold text-finance-text-primary tab-num">
            {volume > 0 ? formatLargeNumber(volume) : 'N/A'}
          </div>
        </div>
        <div>
          <div className="font-medium">Market Cap</div>
          <div className="font-semibold text-finance-text-primary tab-num">
            {marketCap > 0 ? formatLargeNumber(marketCap) : 'N/A'}
          </div>
        </div>
      </div>
    </div>
  );
});

StockCard.displayName = 'StockCard';

export default StockCard;
