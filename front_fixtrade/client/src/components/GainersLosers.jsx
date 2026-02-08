import { getTopGainers, getTopLosers } from '../utils/calculations';
import { formatPercentage, getPriceColorClass, formatCurrency } from '../utils/formatters';

const GainersLosers = ({ stocks }) => {
  const topGainers = getTopGainers(stocks, 5);
  const topLosers = getTopLosers(stocks, 5);

  const renderStockList = (stockList, title, isGainer = true) => (
    <div className="finance-card rounded-lg p-4 md:p-5">
      <div className="flex items-center gap-2 mb-3">
        <h2 className="text-lg md:text-xl font-semibold text-finance-text-primary">
          {title}
        </h2>
        {isGainer ? (
          <svg className="w-5 h-5 text-success-500" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
          </svg>
        ) : (
          <svg className="w-5 h-5 text-danger-500" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M14.707 10.293a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 111.414-1.414L9 12.586V5a1 1 0 012 0v7.586l2.293-2.293a1 1 0 011.414 0z" clipRule="evenodd" />
          </svg>
        )}
      </div>

      {stockList.length === 0 ? (
        <p className="text-finance-text-secondary text-sm">No data available</p>
      ) : (
        <div className="space-y-2">
          {stockList.map((stock, index) => {
            const colorClass = getPriceColorClass(stock.percentChange);
            return (
              <div
                key={stock.id || index}
                className="flex items-center justify-between p-3 rounded bg-finance-bg border border-finance-border hover:border-primary-500/30 transition-colors"
              >
                <div className="flex items-center gap-3 flex-1 min-w-0">
                  <div className="flex-shrink-0 w-7 h-7 rounded bg-primary-500/10 flex items-center justify-center text-xs font-semibold text-primary-400">
                    {index + 1}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-finance-text-primary truncate text-sm">
                      {stock.stockName}
                    </div>
                    <div className="text-xs text-finance-text-secondary">
                      {stock.ticker}
                    </div>
                  </div>
                </div>
                <div className="text-right ml-3">
                  <div className="font-semibold text-finance-text-primary text-sm tab-num">
                    {formatCurrency(stock.last)}
                  </div>
                  <div className={`text-xs font-semibold ${colorClass} tab-num`}>
                    {formatPercentage(stock.percentChange)}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );

  return (
    <div className="mt-6 mb-6">
      <h2 className="text-xl md:text-2xl font-semibold text-finance-text-primary mb-4">
        Market Movers
      </h2>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {renderStockList(topGainers, 'Top Gainers', true)}
        {renderStockList(topLosers, 'Top Losers', false)}
      </div>
    </div>
  );
};

export default GainersLosers;
