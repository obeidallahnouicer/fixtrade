import { useQuery } from '@tanstack/react-query';
import { predictStockPrice } from '../services/api';

/**
 * Custom hook to fetch stock price predictions for a single stock
 * @param {string} symbol - Stock ticker symbol
 * @param {number} horizonDays - Number of days to predict (default: 1)
 * @param {Object} options - React Query options
 * @returns {Object} Query result with prediction data
 */
export const useStockPrediction = (symbol, horizonDays = 1, options = {}) => {
  const queryResult = useQuery({
    queryKey: ['stockPrediction', symbol, horizonDays],
    queryFn: async () => {
      if (!symbol || symbol === 'N/A') {
        throw new Error('Invalid symbol');
      }
      return await predictStockPrice(symbol, horizonDays);
    },
    enabled: !!symbol && symbol !== 'N/A', // Only fetch if symbol is valid
    staleTime: 60000, // Consider predictions fresh for 1 minute
    retry: 1, // Only retry once for predictions
    retryDelay: 1000,
    ...options,
  });

  return {
    prediction: queryResult.data?.predictions?.[0] || null, // Get first day prediction
    predictions: queryResult.data?.predictions || [],
    isLoading: queryResult.isLoading,
    isError: queryResult.isError,
    error: queryResult.error,
  };
};

/**
 * Custom hook to fetch predictions for multiple stocks
 * @param {Array<string>} symbols - Array of stock ticker symbols
 * @param {number} horizonDays - Number of days to predict (default: 1)
 * @returns {Object} Map of symbol to prediction data
 */
export const useMultipleStockPredictions = (symbols, horizonDays = 1) => {
  // TODO: Implement using @tanstack/react-query's useQueries hook
  // For now, we'll use individual queries approach
  console.log('Multiple predictions not yet implemented', symbols, horizonDays);
  return null;
};
