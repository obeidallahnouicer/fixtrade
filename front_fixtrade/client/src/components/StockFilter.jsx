import { useState, useMemo } from 'react';
import { formatPercentage, getPriceColorClass } from '../utils/formatters';

const StockFilter = ({ stocks, tunindexPerformance, onFilterChange, onStockSelect }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [performanceFilter, setPerformanceFilter] = useState('all'); // all, outperforming, underperforming
  const [sortBy, setSortBy] = useState('performance'); // performance, name, volume, marketCap
  const [showFilters, setShowFilters] = useState(false);
  const [selectedStockId, setSelectedStockId] = useState(null);

  // Filter and sort stocks
  const filteredStocks = useMemo(() => {
    let filtered = stocks.filter(stock => {
      // Search filter
      const matchesSearch = searchTerm === '' || 
        (stock.stockName && stock.stockName.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (stock.ticker && stock.ticker.toLowerCase().includes(searchTerm.toLowerCase()));

      // Performance filter vs TUNINDEX
      const stockPerformance = stock.percentChange || 0;
      const indexPerformance = tunindexPerformance || 0;
      
      let matchesPerformance = true;
      if (performanceFilter === 'outperforming') {
        matchesPerformance = stockPerformance > indexPerformance;
      } else if (performanceFilter === 'underperforming') {
        matchesPerformance = stockPerformance < indexPerformance;
      }

      return matchesSearch && matchesPerformance;
    });

    // Sort
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'performance':
          return (b.percentChange || 0) - (a.percentChange || 0);
        case 'name':
          return (a.stockName || '').localeCompare(b.stockName || '');
        case 'volume':
          return (parseFloat(b.volume) || 0) - (parseFloat(a.volume) || 0);
        case 'marketCap':
          return (parseFloat(b.caps) || 0) - (parseFloat(a.caps) || 0);
        default:
          return 0;
      }
    });

    return filtered;
  }, [stocks, searchTerm, performanceFilter, sortBy, tunindexPerformance]);

  // Statistics
  const stats = useMemo(() => {
    const outperforming = stocks.filter(s => (s.percentChange || 0) > (tunindexPerformance || 0)).length;
    const underperforming = stocks.filter(s => (s.percentChange || 0) < (tunindexPerformance || 0)).length;
    const matching = stocks.filter(s => Math.abs((s.percentChange || 0) - (tunindexPerformance || 0)) < 0.1).length;

    return { outperforming, underperforming, matching, total: stocks.length };
  }, [stocks, tunindexPerformance]);

  const handleFilterChange = () => {
    if (onFilterChange) {
      onFilterChange(filteredStocks);
    }
  };

  const handleStockSelect = (stock) => {
    if (selectedStockId === stock.id) {
      // Deselect if clicking the same stock
      setSelectedStockId(null);
      if (onStockSelect) onStockSelect(null);
    } else {
      setSelectedStockId(stock.id);
      if (onStockSelect) onStockSelect(stock);
    }
  };

  return (
    <div className="mb-6">
      {/* Search and Filter Bar */}
      <div className="finance-card rounded-lg p-4 mb-4">
        <div className="flex flex-col md:flex-row gap-3">
          {/* Search Input */}
          <div className="flex-1">
            <div className="relative">
              <svg 
                className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-finance-text-secondary" 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                type="text"
                placeholder="Search stocks by name or ticker..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 bg-finance-bg border border-finance-border rounded text-finance-text-primary placeholder-finance-text-secondary focus:outline-none focus:ring-1 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>
          </div>

          {/* Filter Toggle Button */}
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="px-5 py-2.5 bg-finance-bg hover:bg-[#2a2e39] border border-finance-border rounded font-medium text-finance-text-primary transition-colors flex items-center gap-2 justify-center"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
            </svg>
            Filters
          </button>
        </div>

        {/* Expanded Filters */}
        {showFilters && (
          <div className="overflow-hidden">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4 pt-4 border-t border-finance-border">
              {/* Performance Filter */}
              <div>
                <label className="block text-xs font-medium text-finance-text-secondary mb-2">
                  vs TUNINDEX
                </label>
                <div className="flex gap-2">
                  {[
                    { value: 'all', label: 'All', icon: '📊' },
                    { value: 'outperforming', label: 'Outperforming', icon: '📈' },
                    { value: 'underperforming', label: 'Underperforming', icon: '📉' }
                  ].map(option => (
                    <button
                      key={option.value}
                      onClick={() => {
                        setPerformanceFilter(option.value);
                        handleFilterChange();
                      }}
                      className={`flex-1 px-3 py-2 rounded text-xs font-medium transition-colors ${
                        performanceFilter === option.value
                          ? 'bg-primary-500 text-white'
                          : 'bg-finance-bg border border-finance-border text-finance-text-primary hover:bg-[#2a2e39]'
                      }`}
                    >
                      <span className="mr-1">{option.icon}</span>
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Sort By */}
              <div>
                <label className="block text-xs font-medium text-finance-text-secondary mb-2">
                  Sort By
                </label>
                <select
                  value={sortBy}
                  onChange={(e) => {
                    setSortBy(e.target.value);
                    handleFilterChange();
                  }}
                  className="w-full px-3 py-2 bg-finance-bg border border-finance-border rounded text-finance-text-primary text-sm focus:outline-none focus:ring-1 focus:ring-primary-500 focus:border-primary-500"
                >
                  <option value="performance">Performance Change</option>
                  <option value="name">Stock Name</option>
                  <option value="volume">Trading Volume</option>
                  <option value="marketCap">Market Cap</option>
                </select>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Statistics Bar */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
        <div className="finance-card rounded p-3">
          <div className="text-xs text-finance-text-secondary mb-1">Total Stocks</div>
          <div className="text-lg font-semibold text-finance-text-primary tab-num">{stats.total}</div>
        </div>
        <div className="finance-card rounded p-3">
          <div className="text-xs text-finance-text-secondary mb-1">Outperforming</div>
          <div className="text-lg font-semibold text-success-500 tab-num">{stats.outperforming}</div>
        </div>
        <div className="finance-card rounded p-3">
          <div className="text-xs text-finance-text-secondary mb-1">Underperforming</div>
          <div className="text-lg font-semibold text-danger-500 tab-num">{stats.underperforming}</div>
        </div>
        <div className="finance-card rounded p-3">
          <div className="text-xs text-finance-text-secondary mb-1">Filtered Results</div>
          <div className="text-lg font-semibold text-primary-400 tab-num">{filteredStocks.length}</div>
        </div>
      </div>

      {/* Comparison List */}
      {searchTerm && filteredStocks.length > 0 && (
        <div className="finance-card rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-base font-semibold text-finance-text-primary">
              Performance Comparison
            </h3>
            {selectedStockId && (
              <button
                onClick={() => handleStockSelect({ id: selectedStockId })}
                className="text-xs text-primary-400 hover:text-primary-300 flex items-center gap-1"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
                Clear Selection
              </button>
            )}
          </div>
          <div className="space-y-2 max-h-[300px] overflow-y-auto">
            {filteredStocks.slice(0, 10).map((stock, index) => {
              const stockPerf = stock.percentChange || 0;
              const indexPerf = tunindexPerformance || 0;
              const difference = stockPerf - indexPerf;
              const isOutperforming = difference > 0;
              const isSelected = selectedStockId === stock.id;

              return (
                <div
                  key={stock.id || index}
                  className={`flex items-center justify-between p-3 rounded border transition-all cursor-pointer ${
                    isSelected 
                      ? 'bg-primary-500/10 border-primary-500' 
                      : 'bg-finance-bg border-finance-border hover:border-primary-500/30'
                  }`}
                  onClick={() => handleStockSelect(stock)}
                >
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    <div className={`w-8 h-8 rounded flex items-center justify-center text-xs font-semibold ${
                      isSelected ? 'bg-primary-500 text-white' : 'bg-finance-border text-finance-text-secondary'
                    }`}>
                      {isSelected ? '✓' : '📊'}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="font-medium text-finance-text-primary truncate text-sm">
                        {stock.stockName || 'Unknown'}
                      </div>
                      <div className="text-xs text-finance-text-secondary">
                        {stock.ticker || 'N/A'}
                      </div>
                    </div>
                  </div>
                  <div className="text-right ml-4">
                    <div className={`text-sm font-semibold ${getPriceColorClass(stockPerf)} tab-num`}>
                      {formatPercentage(stockPerf)}
                    </div>
                    <div className={`text-xs font-medium ${
                      isOutperforming ? 'text-[#10b981]' : 'text-[#ef4444]'
                    } tab-num`}>
                      {isOutperforming ? '+' : ''}{difference.toFixed(2)}% vs Index
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default StockFilter;
