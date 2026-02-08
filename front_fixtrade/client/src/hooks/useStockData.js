import { useQuery } from '@tanstack/react-query';
import { fetchStocks } from '../services/api';
import { calculatePercentageChange } from '../utils/calculations';

/**
 * Custom hook to fetch and manage stock data with auto-refresh
 * @param {Object} options - React Query options
 * @returns {Object} Query result with stocks data
 */
export const useStockData = (options = {}) => {
  const queryResult = useQuery({
    queryKey: ['stocks'],
    queryFn: async () => {
      const data = await fetchStocks();
      
      // Validate response structure
      if (!data || !data.markets) {
        console.error('Invalid API response:', data);
        throw new Error('Invalid API response structure');
      }

      console.log('Raw API data:', data); // Debug log
      console.log('First stock sample:', data.markets[0]); // Debug log

      // Transform data: add percentage change to each stock
      const stocksWithPercentage = data.markets.map(stock => {
        // Handle nested referentiel structure if it exists
        const stockName = stock.stockName || stock.referentiel?.stockName || stock.arabName || 'Unknown';
        const ticker = stock.ticker || stock.referentiel?.ticker || 'N/A';
        
        return {
          ...stock,
          stockName,
          ticker,
          percentChange: calculatePercentageChange(stock),
        };
      });

      console.log('Transformed stocks sample:', stocksWithPercentage[0]); // Debug log

      return {
        ...data,
        markets: stocksWithPercentage,
      };
    },
    refetchInterval: 30000, // Refetch every 30 seconds
    staleTime: 25000, // Consider data stale after 25 seconds
    retry: 3, // Retry failed requests 3 times
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000), // Exponential backoff
    refetchOnWindowFocus: true,
    refetchOnReconnect: true,
    ...options,
  });

  return {
    stocks: queryResult.data?.markets || [],
    rawData: queryResult.data,
    isLoading: queryResult.isLoading,
    isError: queryResult.isError,
    error: queryResult.error,
    dataUpdatedAt: queryResult.dataUpdatedAt,
    refetch: queryResult.refetch,
    isRefetching: queryResult.isRefetching,
  };
};
