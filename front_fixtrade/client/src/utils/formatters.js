/**
 * Format a number as currency (TND)
 * @param {number} value - The value to format
 * @returns {string} Formatted currency string
 */
export const formatCurrency = (value) => {
  if (value === null || value === undefined || isNaN(value)) {
    return '0.00 TND';
  }
  const num = parseFloat(value);
  if (num === 0) {
    return '0.00 TND';
  }
  return `${num.toFixed(2)} TND`;
};

/**
 * Format a number as percentage with sign
 * @param {number} value - The percentage value
 * @returns {string} Formatted percentage string
 */
export const formatPercentage = (value) => {
  if (value === null || value === undefined || isNaN(value)) {
    return '0.00%';
  }
  const num = parseFloat(value);
  const sign = num > 0 ? '+' : '';
  return `${sign}${num.toFixed(2)}%`;
};

/**
 * Format large numbers with K/M suffixes
 * @param {number} value - The value to format
 * @returns {string} Formatted number string
 */
export const formatLargeNumber = (value) => {
  if (value === null || value === undefined || isNaN(value)) {
    return '0';
  }
  
  const num = parseFloat(value);
  
  if (num === 0) {
    return '0';
  }
  
  if (num >= 1000000) {
    return `${(num / 1000000).toFixed(2)}M`;
  }
  
  if (num >= 1000) {
    return `${(num / 1000).toFixed(2)}K`;
  }
  
  return num.toFixed(0);
};

/**
 * Get relative time string
 * @param {Date|string|number} timestamp - The timestamp to format
 * @returns {string} Relative time string
 */
export const getRelativeTime = (timestamp) => {
  if (!timestamp) return 'Unknown';
  
  const now = new Date();
  const time = new Date(timestamp);
  const diffInSeconds = Math.floor((now - time) / 1000);
  
  if (diffInSeconds < 60) {
    return 'Just now';
  }
  
  const diffInMinutes = Math.floor(diffInSeconds / 60);
  if (diffInMinutes < 60) {
    return `${diffInMinutes} ${diffInMinutes === 1 ? 'minute' : 'minutes'} ago`;
  }
  
  const diffInHours = Math.floor(diffInMinutes / 60);
  if (diffInHours < 24) {
    return `${diffInHours} ${diffInHours === 1 ? 'hour' : 'hours'} ago`;
  }
  
  const diffInDays = Math.floor(diffInHours / 24);
  return `${diffInDays} ${diffInDays === 1 ? 'day' : 'days'} ago`;
};

/**
 * Get color class based on value
 * @param {number} value - The value to check
 * @returns {string} Tailwind color class
 */
export const getPriceColorClass = (value) => {
  if (value > 0) return 'metric-up';
  if (value < 0) return 'metric-down';
  return 'metric-neutral';
};

/**
 * Format time string
 * @param {string} time - Time string to format
 * @returns {string} Formatted time
 */
export const formatTime = (time) => {
  if (!time) return '';
  return time;
};
