# Tunisian Stock Market Dashboard

A modern, real-time stock market dashboard displaying BVMT (Bourse des Valeurs Mobilières de Tunis) data with an Apple-inspired UI. Built with React, Vite, Tailwind CSS, and Framer Motion.

## Features

- **TUNINDEX Display**: Market-cap weighted calculation of the main Tunisian stock index
- **Real-time Updates**: Auto-refresh every 30 seconds with React Query
- **All Stocks Grid**: Comprehensive view of all listed stocks with key metrics
- **Top Gainers/Losers**: Top 5 performers in each category
- **Stock Details**: Detailed modal view with order book, trading volume, and price history
- **Apple-Style UI**: Glassmorphism effects, smooth animations, and elegant design
- **Mobile Responsive**: Optimized for all screen sizes from mobile to desktop
- **Dark Mode Ready**: Supports system dark mode preferences

## Tech Stack

### Frontend (Client)
- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first styling
- **Framer Motion** - Smooth animations
- **React Query** - Data fetching and caching
- **Axios** - HTTP client

### Backend (Server)
- **Node.js** - Runtime
- **Express** - Web server
- **CORS** - Cross-origin resource sharing

## Project Structure

```
front_fixtrade/
├── client/                  # React frontend
│   ├── src/
│   │   ├── components/      # React components
│   │   │   ├── Dashboard.jsx
│   │   │   ├── IndexCard.jsx
│   │   │   ├── StockCard.jsx
│   │   │   ├── GainersLosers.jsx
│   │   │   ├── StockDetail.jsx
│   │   │   └── Skeleton.jsx
│   │   ├── hooks/           # Custom React hooks
│   │   │   ├── useStockData.js
│   │   │   └── useCalculateTunindex.js
│   │   ├── services/        # API services
│   │   │   └── api.js
│   │   ├── utils/           # Utility functions
│   │   │   ├── formatters.js
│   │   │   └── calculations.js
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── .env
│   ├── tailwind.config.js
│   └── package.json
├── server/                  # Express proxy server
│   ├── index.js
│   ├── .env
│   └── package.json
└── package.json             # Root package.json

```

## Installation

### Prerequisites
- Node.js 18+ and npm

### Setup

1. **Clone or navigate to the project directory**
   ```bash
   cd front_fixtrade
   ```

2. **Install root dependencies**
   ```bash
   npm install
   ```

3. **Environment variables are already configured**
   - `client/.env` - Points to local proxy server
   - `server/.env` - Contains BVMT API endpoint

## Running the Application

### Development Mode (Recommended)

Run both client and server concurrently:
```bash
npm run dev
```

This starts:
- **Server**: http://localhost:5000 (proxy)
- **Client**: http://localhost:5173 (React app)

### Run Individually

**Server only:**
```bash
npm run dev:server
```

**Client only:**
```bash
npm run dev:client
```

### Production Build

Build the client:
```bash
npm run build
```

The optimized build will be in `client/dist/`

## Architecture

### Data Flow

1. **React App** requests data from custom hooks (`useStockData`)
2. **React Query** manages caching, refetching, and state
3. **Axios** sends HTTP request to local proxy server
4. **Express server** forwards request to BVMT API
5. **Data flows back** through the chain with transformations

### CORS Solution

The BVMT API blocks direct browser requests due to CORS policy. Our Express proxy server:
- Acts as intermediary between frontend and BVMT API
- Adds proper CORS headers
- Handles timeouts and errors gracefully

### TUNINDEX Calculation

Since the API doesn't provide TUNINDEX directly, we calculate it:
1. Filter stocks with `valGroup = "11"` (primary market)
2. Calculate weighted average: `Σ(price × marketCap) / Σ(marketCap)`
3. Compare to previous close for percentage change

## Key Features Implementation

### Auto-Refresh
- React Query refetches data every 30 seconds
- Exponential backoff on errors
- Pauses when tab is not focused (browser optimization)

### Animations
- Framer Motion for smooth transitions
- Staggered list animations
- Number counter for index value
- Modal enter/exit animations

### Responsive Design
- Mobile-first approach
- Tailwind breakpoints: `sm`, `md`, `lg`, `xl`
- Touch-friendly tap targets (44px minimum)
- Conditional glassmorphism (reduced on mobile for performance)

### Error Handling
- Network errors: Retry with user feedback
- Timeouts: 10-second limit with fallback
- API errors: Graceful degradation
- Loading states: Skeleton screens

## Customization

### Polling Interval

Edit `client/src/hooks/useStockData.js`:
```javascript
refetchInterval: 30000, // Change to desired milliseconds
```

### API Endpoint

Edit `server/.env`:
```
BVMT_API_URL=https://www.bvmt.com.tn/rest_api/rest/market/groups/11,12,52,95,99
```

### Theme Colors

Edit `client/tailwind.config.js`:
```javascript
colors: {
  appleBlue: '#0071e3',
  appleGreen: '#30d158',
  appleRed: '#ff3b30',
}
```

## Browser Compatibility

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+ (with `-webkit-backdrop-filter`)
- Mobile browsers (iOS Safari, Chrome Mobile)

**Note**: Glassmorphism requires modern browser support for `backdrop-filter`

## Performance Optimizations

- **React.memo** on StockCard to prevent unnecessary re-renders
- **useMemo** for expensive calculations (TUNINDEX, sorting)
- **Code splitting** via Vite
- **Tailwind purge** removes unused CSS in production
- **Reduced animations** on mobile devices

## Troubleshooting

### Server won't start
- Check if port 5000 is available
- Verify Node.js version (18+)
- Run `npm install` in server folder

### Client shows CORS errors
- Ensure server is running on port 5000
- Check `client/.env` has correct API URL
- Verify proxy server logs for errors

### No data loading
- Check BVMT API accessibility: https://www.bvmt.com.tn/rest_api/rest/market/groups/11,12,52,95,99
- Verify server logs for API errors
- Check browser console for network errors

### Animations laggy on mobile
- This is expected - reduced blur effects are intentional
- Glassmorphism is performance-intensive
- Consider further reducing `backdrop-blur` values

## Development Tips

### Adding New Components
```bash
# Create in client/src/components/
touch client/src/components/NewComponent.jsx
```

### Testing API Locally
```bash
# Direct API test
curl http://localhost:5000/api/stocks
```

### Viewing React Query DevTools
Add to `client/src/App.jsx`:
```javascript
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
// Add <ReactQueryDevtools /> in render
```

## Future Enhancements

- [ ] Historical price charts (Chart.js or Recharts)
- [ ] Search/filter stocks functionality
- [ ] Favorites/watchlist with localStorage
- [ ] Dark mode toggle (currently auto-detect)
- [ ] Export data to CSV
- [ ] Push notifications for price alerts
- [ ] Multi-language support (Arabic/French)
- [ ] WebSocket integration (if BVMT adds support)

## License

ISC

## Credits

- **BVMT API**: Bourse des Valeurs Mobilières de Tunis
- **UI Inspiration**: Apple Design Guidelines
- **Icons**: Heroicons (embedded SVG)

---

**Built with ❤️ for the Tunisian financial community**
