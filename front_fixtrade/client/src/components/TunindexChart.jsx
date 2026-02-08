import { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Area, AreaChart, ComposedChart } from 'recharts';
import { chartConfig, chartColors } from '../utils/chartConfig';

const TunindexChart = ({ indexData, comparisonStock, historicalData = [] }) => {
  const [timeRange, setTimeRange] = useState('1D');

  // Generate historical data for demonstration - memoized to avoid impure function calls during render
  const chartData = useMemo(() => {
    if (historicalData && historicalData.length > 0) {
      return historicalData;
    }

    // Generate realistic intraday data points
    const points = timeRange === '1D' ? 24 : timeRange === '1W' ? 7 : 30;
    const baseValue = indexData?.indexValue || 8000;
    const stockBasePrice = comparisonStock ? parseFloat(comparisonStock.last || comparisonStock.close || 10) : 0;
    const data = [];

    // Use a more realistic volatility pattern
    let currentIndexValue = baseValue;
    let currentStockValue = stockBasePrice;

    for (let i = 0; i < points; i++) {
      // More realistic random walk with mean reversion
      const indexVariance = (Math.random() - 0.5) * (baseValue * 0.01); // 1% max variance
      const stockVariance = (Math.random() - 0.5) * (stockBasePrice * 0.02); // 2% max variance
      
      currentIndexValue += indexVariance;
      currentStockValue += stockVariance;
      
      let time;
      if (timeRange === '1D') {
        time = `${9 + Math.floor(i / 2)}:${i % 2 === 0 ? '00' : '30'}`;
      } else if (timeRange === '1W') {
        time = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][i];
      } else {
        time = `Day ${i + 1}`;
      }

      const dataPoint = {
        time,
        indexValue: parseFloat(currentIndexValue.toFixed(2)),
        indexChange: parseFloat(((currentIndexValue - baseValue) / baseValue * 100).toFixed(2)),
      };

      // Add comparison stock data if available
      if (comparisonStock) {
        dataPoint.stockValue = parseFloat(currentStockValue.toFixed(2));
        dataPoint.stockChange = parseFloat(((currentStockValue - stockBasePrice) / stockBasePrice * 100).toFixed(2));
      }

      data.push(dataPoint);
    }

    return data;
  }, [timeRange, indexData, comparisonStock, historicalData]);

  const currentValue = indexData?.indexValue || 0;
  const percentageChange = indexData?.percentageChange || 0;
  const isPositive = percentageChange >= 0;

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="finance-card p-3 border border-[#3a3e49]">
          <p className="text-xs text-gray-400 mb-2">
            {data.time}
          </p>
          <div className="space-y-2">
            <div>
              <p className="text-xs text-gray-500">TUNINDEX</p>
              <p className="text-base font-bold text-white tab-num">
                {data.indexValue?.toFixed(2)}
              </p>
              <p className={`text-sm font-semibold tab-num ${
                data.indexChange >= 0 ? 'metric-up' : 'metric-down'
              }`}>
                {data.indexChange >= 0 ? '+' : ''}{data.indexChange}%
              </p>
            </div>
            {comparisonStock && data.stockValue && (
              <div className="pt-2 border-t border-[#2a2e39]">
                <p className="text-xs text-gray-500">{comparisonStock.stockName || comparisonStock.ticker}</p>
                <p className="text-base font-bold text-primary-400 tab-num">
                  {data.stockValue?.toFixed(2)}
                </p>
                <p className={`text-sm font-semibold tab-num ${
                  data.stockChange >= 0 ? 'metric-up' : 'metric-down'
                }`}>
                  {data.stockChange >= 0 ? '+' : ''}{data.stockChange}%
                </p>
              </div>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="finance-card p-6 mb-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6">
        <div className="flex-1">
          <h2 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-2">
            {comparisonStock ? 'Performance Comparison' : 'Index Performance'}
          </h2>
          <div className="flex flex-col md:flex-row md:items-center gap-4">
            <div className="flex items-baseline gap-3">
              <span className="text-2xl font-bold text-white tab-num">
                {currentValue.toFixed(2)}
              </span>
              <span className={`text-base font-semibold tab-num ${
                isPositive ? 'metric-up' : 'metric-down'
              }`}>
                {isPositive ? '+' : ''}{percentageChange.toFixed(2)}%
              </span>
              <span className="text-xs text-gray-500">TUNINDEX</span>
            </div>
            {comparisonStock && (
              <div className="flex items-baseline gap-3">
                <span className="text-2xl font-bold text-primary-400 tab-num">
                  {parseFloat(comparisonStock.last || comparisonStock.close || 0).toFixed(2)}
                </span>
                <span className={`text-base font-semibold tab-num ${
                  (comparisonStock.percentChange || 0) >= 0 ? 'metric-up' : 'metric-down'
                }`}>
                  {(comparisonStock.percentChange || 0) >= 0 ? '+' : ''}{(comparisonStock.percentChange || 0).toFixed(2)}%
                </span>
                <span className="text-xs text-gray-500">{comparisonStock.stockName || comparisonStock.ticker}</span>
              </div>
            )}
          </div>
        </div>

        {/* Time Range Selector */}
        <div className="flex gap-2 mt-4 md:mt-0">
          {['1D', '1W', '1M'].map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-4 py-1.5 rounded text-sm font-medium transition-all ${
                timeRange === range
                  ? 'bg-primary-500 text-white'
                  : 'bg-[#2a2e39] text-gray-400 hover:bg-[#3a3e49] hover:text-gray-300'
              }`}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      {/* Chart */}
      <div className="w-full h-[300px] md:h-[400px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData}>
            <defs>
              <linearGradient id="colorIndex" x1="0" y1="0" x2="0" y2="1">
                <stop 
                  offset="5%" 
                  stopColor={isPositive ? '#10b981' : '#ef4444'} 
                  stopOpacity={0.3}
                />
                <stop 
                  offset="95%" 
                  stopColor={isPositive ? '#10b981' : '#ef4444'} 
                  stopOpacity={0.1}
                />
              </linearGradient>
              <linearGradient id="colorStock" x1="0" y1="0" x2="0" y2="1">
                <stop 
                  offset="5%" 
                  stopColor="#3b82f6" 
                  stopOpacity={0.2}
                />
                <stop 
                  offset="95%" 
                  stopColor="#3b82f6" 
                  stopOpacity={0.05}
                />
              </linearGradient>
            </defs>
            <CartesianGrid {...chartConfig.cartesianGrid} />
            <XAxis 
              dataKey="time" 
              {...chartConfig.axis}
            />
            <YAxis 
              yAxisId="left"
              {...chartConfig.axis}
              domain={['auto', 'auto']}
            />
            {comparisonStock && (
              <YAxis 
                yAxisId="right"
                orientation="right"
                stroke={chartColors.primary}
                style={{ fontSize: '12px' }}
                tick={{ fill: chartColors.primary }}
                domain={['auto', 'auto']}
              />
            )}
            <Tooltip content={<CustomTooltip />} />
            <Legend 
              {...chartConfig.legend}
            />
            <Area
              yAxisId="left"
              type="monotone"
              dataKey="indexValue"
              name="TUNINDEX"
              stroke={isPositive ? chartColors.success : chartColors.danger}
              strokeWidth={2}
              fill="url(#colorIndex)"
            />
            {comparisonStock && (
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="stockValue"
                name={comparisonStock.stockName || comparisonStock.ticker}
                stroke={chartColors.primary}
                strokeWidth={2}
                dot={false}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6 pt-6 border-t border-[#2a2e39]">
        <div>
          <div className="text-xs text-gray-500 mb-1">Index High</div>
          <div className="text-base font-semibold text-gray-300 tab-num">
            {Math.max(...chartData.map(d => d.indexValue || 0)).toFixed(2)}
          </div>
        </div>
        <div>
          <div className="text-xs text-gray-500 mb-1">Index Low</div>
          <div className="text-base font-semibold text-gray-300 tab-num">
            {Math.min(...chartData.map(d => d.indexValue || 0)).toFixed(2)}
          </div>
        </div>
        <div>
          <div className="text-xs text-gray-500 mb-1">Index Avg</div>
          <div className="text-base font-semibold text-gray-300 tab-num">
            {(chartData.reduce((sum, d) => sum + (d.indexValue || 0), 0) / chartData.length).toFixed(2)}
          </div>
        </div>
        <div>
          <div className="text-xs text-gray-500 mb-1">Volatility</div>
          <div className="text-base font-semibold text-gray-300 tab-num">
            {(Math.max(...chartData.map(d => Math.abs(d.indexChange || 0)))).toFixed(2)}%
          </div>
        </div>
      </div>
    </div>
  );
};

export default TunindexChart;
