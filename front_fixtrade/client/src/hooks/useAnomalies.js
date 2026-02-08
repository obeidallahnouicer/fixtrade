import { useQuery } from '@tanstack/react-query';
import { getRecentAnomalies, detectAnomalies } from '../services/api';

/**
 * Fetch recent anomaly alerts from the backend.
 */
export const useRecentAnomalies = ({ symbol = null, limit = 50, hoursBack = 24 } = {}, options = {}) => {
  const query = useQuery({
    queryKey: ['anomalies', 'recent', symbol, limit, hoursBack],
    queryFn: () => getRecentAnomalies({ symbol, limit, hoursBack }),
    staleTime: 30000,
    refetchInterval: 60000, // auto-refresh every 60s
    retry: 1,
    ...options,
  });

  return {
    anomalies: query.data?.anomalies || [],
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    refetch: query.refetch,
  };
};

/**
 * Detect anomalies for a specific symbol on-demand.
 */
export const useDetectAnomalies = (symbol, options = {}) => {
  const query = useQuery({
    queryKey: ['anomalies', 'detect', symbol],
    queryFn: () => detectAnomalies(symbol),
    enabled: !!symbol && symbol !== 'N/A',
    staleTime: 60000,
    retry: 1,
    ...options,
  });

  return {
    anomalies: query.data?.anomalies || [],
    isLoading: query.isLoading,
    isError: query.isError,
  };
};
