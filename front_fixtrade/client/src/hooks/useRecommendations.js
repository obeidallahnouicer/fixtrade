import { useQuery } from '@tanstack/react-query';
import {
  getTradeRecommendation,
  getAIRecommendations,
  getRecommendationExplanation,
} from '../services/api';

/**
 * Fetch a single trade recommendation for a stock.
 */
export const useTradeRecommendation = (symbol, portfolioId = 'default', options = {}) => {
  const query = useQuery({
    queryKey: ['recommendation', symbol, portfolioId],
    queryFn: () => getTradeRecommendation(symbol, portfolioId),
    enabled: !!symbol && symbol !== 'N/A',
    staleTime: 60000,
    retry: 1,
    ...options,
  });

  return {
    recommendation: query.data,
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
  };
};

/**
 * Fetch AI daily recommendations (top N).
 */
export const useAIRecommendations = (portfolioId = 'default', topN = 10, options = {}) => {
  const query = useQuery({
    queryKey: ['ai-recommendations', portfolioId, topN],
    queryFn: () => getAIRecommendations({ portfolioId, topN }),
    staleTime: 120000,
    retry: 1,
    ...options,
  });

  return {
    recommendations: query.data || [],
    isLoading: query.isLoading,
    isError: query.isError,
  };
};

/**
 * Fetch AI explanation for a specific stock recommendation.
 */
export const useRecommendationExplanation = (symbol, portfolioId = 'default', options = {}) => {
  const query = useQuery({
    queryKey: ['recommendation-explain', symbol, portfolioId],
    queryFn: () => getRecommendationExplanation(symbol, portfolioId),
    enabled: !!symbol,
    staleTime: 300000,
    retry: 1,
    ...options,
  });

  return {
    explanation: query.data?.explanation || null,
    isLoading: query.isLoading,
    isError: query.isError,
  };
};
