import { useQuery } from '@tanstack/react-query';
import { getStockSentiment } from '../services/api';

/**
 * Fetch sentiment analysis for a given stock symbol.
 * Returns { score, sentiment, articleCount }.
 */
export const useSentiment = (symbol, options = {}) => {
  const query = useQuery({
    queryKey: ['sentiment', symbol],
    queryFn: () => getStockSentiment(symbol),
    enabled: !!symbol && symbol !== 'N/A',
    staleTime: 120000,
    retry: 1,
    ...options,
  });

  return {
    sentiment: query.data,
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
  };
};
