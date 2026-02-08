import { formatPercentage, getPriceColorClass, getRelativeTime } from '../utils/formatters';

const IndexCard = ({ value, percentageChange, change, lastUpdated }) => {
  const colorClass = getPriceColorClass(percentageChange);

  return (
    <div className="finance-card rounded-lg p-4 md:p-6 mb-4 border-b-2 border-finance-border">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        <div className="flex-1">
          <h1 className="text-lg md:text-xl font-semibold text-finance-text-primary mb-1">
            TUNINDEX
          </h1>
          <p className="text-xs text-finance-text-secondary">
            Bourse de Tunis Main Index
          </p>
        </div>

        <div className="flex-1 text-left md:text-right">
          <div className="text-3xl md:text-4xl font-bold text-finance-text-primary mb-1 tab-num">
            {value ? value.toFixed(2) : '0.00'}
          </div>

          <div className={`flex items-center gap-2 justify-start md:justify-end text-base md:text-lg font-semibold ${colorClass}`}>
            {percentageChange > 0 && (
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
              </svg>
            )}
            {percentageChange < 0 && (
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M14.707 10.293a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 111.414-1.414L9 12.586V5a1 1 0 012 0v7.586l2.293-2.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
            )}
            <span className="tab-num">{formatPercentage(percentageChange)}</span>
            <span className="text-sm tab-num">({change > 0 ? '+' : ''}{change ? change.toFixed(2) : '0.00'})</span>
          </div>

          {lastUpdated && (
            <p className="text-xs text-finance-text-secondary mt-1">
              Updated {getRelativeTime(lastUpdated)}
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default IndexCard;
