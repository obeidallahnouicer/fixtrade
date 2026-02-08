import { useQuery } from '@tanstack/react-query';
import { predictVolume, predictLiquidity } from '../services/api';

/**
 * Fetch volume predictions for a stock.
 */
export const useVolumePrediction = (symbol, horizonDays = 5, options = {}) => {
  const query = useQuery({
    queryKey: ['volume-prediction', symbol, horizonDays],
    queryFn: () => predictVolume(symbol, horizonDays),
    enabled: !!symbol && symbol !== 'N/A',
    staleTime: 120000,
    retry: 1,
    ...options,
  });

  return {
    volumePredictions: query.data?.predictions || [],
    isLoading: query.isLoading,
    isError: query.isError,
  };
};

/**
 * Fetch liquidity predictions for a stock.
 */
export const useLiquidityPrediction = (symbol, horizonDays = 5, options = {}) => {
  const query = useQuery({
    queryKey: ['liquidity-prediction', symbol, horizonDays],
    queryFn: () => predictLiquidity(symbol, horizonDays),
    enabled: !!symbol && symbol !== 'N/A',
    staleTime: 120000,
    retry: 1,
    ...options,
  });

  return {
    liquidityForecasts: query.data?.forecasts || [],
    isLoading: query.isLoading,
    isError: query.isError,
  };
};
