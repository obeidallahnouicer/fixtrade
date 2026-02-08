import { useState } from 'react';
import { useStockData } from '../hooks/useStockData';
import { useCalculateTunindex } from '../hooks/useCalculateTunindex';
import Sidebar from './Sidebar';
import IndexCard from './IndexCard';
import StockCard from './StockCard';
import GainersLosers from './GainersLosers';
import StockDetail from './StockDetail';
import TunindexChart from './TunindexChart';
import StockFilter from './StockFilter';
import Skeleton from './Skeleton';

const Dashboard = () => {
  const { stocks, isLoading, isError, error, dataUpdatedAt } = useStockData();
  const tunindexData = useCalculateTunindex(stocks);
  const [selectedStock, setSelectedStock] = useState(null);
  const [isDetailOpen, setIsDetailOpen] = useState(false);
  const [filteredStocks, setFilteredStocks] = useState([]);
  const [comparisonStock, setComparisonStock] = useState(null);
  const [currentView, setCurrentView] = useState('dashboard');

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

  return (
    <div className="flex min-h-screen bg-[#0f1117]">
      {/* Sidebar */}
      <Sidebar currentView={currentView} onNavigate={setCurrentView} />
      
      {/* Main Content */}
      <div className="flex-1 overflow-auto">
        <div className="p-4 md:p-6 lg:p-8">
          <div className="max-w-[1600px] mx-auto">
        {/* TUNINDEX Card */}
        <IndexCard
          value={tunindexData.indexValue}
          percentageChange={tunindexData.percentageChange}
          change={tunindexData.change}
          lastUpdated={dataUpdatedAt}
        />

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

        {/* All Stocks Grid */}
        <div className="mb-6">
          <h2 className="text-xl md:text-2xl font-semibold text-finance-text-primary mb-4">
            {filteredStocks.length > 0 ? `Filtered Stocks (${displayStocks.length})` : 'All Stocks'}
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

        {/* Gainers and Losers */}
        <GainersLosers stocks={stocks} />

        {/* Stock Detail Modal */}
        <StockDetail
          stock={selectedStock}
          isOpen={isDetailOpen}
          onClose={handleCloseDetail}
        />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
