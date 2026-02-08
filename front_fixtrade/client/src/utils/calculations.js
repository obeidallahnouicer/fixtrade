/**
 * Calculate percentage change for a stock
 * @param {Object} stock - Stock object with change and close properties
 * @returns {number} Percentage change
 */
export const calculatePercentageChange = (stock) => {
  if (!stock) return 0;
  
  // Parse values as floats, handling both string and number inputs
  const currentPrice = parseFloat(stock.last || stock.close || 0);
  const previousClose = parseFloat(stock.close || 0);
  const change = parseFloat(stock.change || 0);
  
  // If we have explicit change and close, calculate from that
  if (previousClose > 0 && change !== 0) {
    return (change / previousClose) * 100;
  }
  
  // Otherwise calculate from current vs previous
  if (previousClose > 0 && currentPrice > 0) {
    return ((currentPrice - previousClose) / previousClose) * 100;
  }
  
  return 0;
};

/**
 * Calculate TUNINDEX from primary market stocks (valGroup = 11)
 * Uses market cap weighted average
 * @param {Array} stocks - Array of stock objects
 * @returns {Object} Object with indexValue and percentageChange
 */
export const calculateTunindex = (stocks) => {
  if (!stocks || !Array.isArray(stocks) || stocks.length === 0) {
    return { indexValue: 0, percentageChange: 0, change: 0, stocks: [] };
  }

  // Filter primary market stocks (valGroup = "11" or 11)
  const primaryStocks = stocks.filter(stock => 
    stock.valGroup === "11" || stock.valGroup === 11
  );
  
  if (primaryStocks.length === 0) {
    return { indexValue: 0, percentageChange: 0, change: 0, stocks: [] };
  }

  // Calculate weighted index value and previous value
  let totalWeightedValue = 0;
  let totalWeightedPreviousValue = 0;
  let totalMarketCap = 0;

  primaryStocks.forEach(stock => {
    // Parse all values as floats to handle string inputs
    const marketCap = parseFloat(stock.caps || stock.marketCap || 0);
    const currentPrice = parseFloat(stock.last || stock.lastPrice || 0);
    const previousPrice = parseFloat(stock.close || stock.previousClose || 0);

    if (marketCap > 0 && currentPrice > 0 && previousPrice > 0) {
      totalWeightedValue += currentPrice * marketCap;
      totalWeightedPreviousValue += previousPrice * marketCap;
      totalMarketCap += marketCap;
    }
  });

  if (totalMarketCap === 0) {
    return { indexValue: 0, percentageChange: 0, change: 0, stocks: primaryStocks };
  }

  const indexValue = totalWeightedValue / totalMarketCap;
  const previousIndexValue = totalWeightedPreviousValue / totalMarketCap;
  
  const change = indexValue - previousIndexValue;
  const percentageChange = previousIndexValue !== 0 
    ? (change / previousIndexValue) * 100 
    : 0;

  return {
    indexValue: parseFloat(indexValue.toFixed(2)),
    percentageChange: parseFloat(percentageChange.toFixed(2)),
    change: parseFloat(change.toFixed(2)),
    stocks: primaryStocks,
  };
};

/**
 * Get top gainers from stocks array
 * @param {Array} stocks - Array of stock objects with percentChange
 * @param {number} count - Number of top gainers to return
 * @returns {Array} Array of top gaining stocks
 */
export const getTopGainers = (stocks, count = 5) => {
  if (!stocks || !Array.isArray(stocks)) {
    return [];
  }

  return [...stocks]
    .sort((a, b) => (b.percentChange || 0) - (a.percentChange || 0))
    .filter(stock => (stock.percentChange || 0) > 0)
    .slice(0, count);
};

/**
 * Get top losers from stocks array
 * @param {Array} stocks - Array of stock objects with percentChange
 * @param {number} count - Number of top losers to return
 * @returns {Array} Array of top losing stocks
 */
export const getTopLosers = (stocks, count = 5) => {
  if (!stocks || !Array.isArray(stocks)) {
    return [];
  }

  return [...stocks]
    .sort((a, b) => (a.percentChange || 0) - (b.percentChange || 0))
    .filter(stock => (stock.percentChange || 0) < 0)
    .slice(0, count);
};

/**
 * Sort stocks by various criteria
 * @param {Array} stocks - Array of stocks
 * @param {string} criteria - Sort criteria (volume, change, marketCap, name)
 * @param {string} order - Sort order (asc, desc)
 * @returns {Array} Sorted array
 */
export const sortStocks = (stocks, criteria = 'volume', order = 'desc') => {
  if (!stocks || !Array.isArray(stocks)) {
    return [];
  }

  const sortedStocks = [...stocks];

  sortedStocks.sort((a, b) => {
    let aValue, bValue;

    switch (criteria) {
      case 'volume':
        aValue = parseFloat(a.volume || 0);
        bValue = parseFloat(b.volume || 0);
        break;
      case 'change':
        aValue = a.percentChange || 0;
        bValue = b.percentChange || 0;
        break;
      case 'marketCap':
        aValue = parseFloat(a.caps || 0);
        bValue = parseFloat(b.caps || 0);
        break;
      case 'name':
        aValue = (a.stockName || '').toLowerCase();
        bValue = (b.stockName || '').toLowerCase();
        return order === 'asc' 
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      default:
        return 0;
    }

    return order === 'asc' ? aValue - bValue : bValue - aValue;
  });

  return sortedStocks;
};
