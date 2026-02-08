import { useMemo } from 'react';
import { calculateTunindex } from '../utils/calculations';

/**
 * Custom hook to calculate TUNINDEX from stock data
 * @param {Array} stocks - Array of stock objects
 * @returns {Object} TUNINDEX data with value and percentage change
 */
export const useCalculateTunindex = (stocks) => {
  const tunindexData = useMemo(() => {
    if (!stocks || stocks.length === 0) {
      return {
        indexValue: 0,
        percentageChange: 0,
        change: 0,
        stocks: [],
      };
    }

    return calculateTunindex(stocks);
  }, [stocks]);

  return tunindexData;
};
