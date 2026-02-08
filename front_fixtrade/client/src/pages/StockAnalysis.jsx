import { useState, useMemo, useEffect } from 'react';
import { useStockData } from '../hooks/useStockData';
import { useStockPrediction } from '../hooks/useStockPredictions';
import { useSentiment } from '../hooks/useSentiment';
import { useTradeRecommendation } from '../hooks/useRecommendations';
import { useVolumePrediction, useLiquidityPrediction } from '../hooks/useVolumeAndLiquidity';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { BarChart3, TrendingUp, Activity, AlertCircle, Droplets, Volume2 } from 'lucide-react';
import { chartConfig, chartColors, lineConfigs } from '../utils/chartConfig';

const StockAnalysis = () => {
  const { stocks, isLoading } = useStockData();
  const [selectedStock, setSelectedStock] = useState('');
  const [horizonDays] = useState(5);

  // Stable "now" used for relative timestamps and deterministic memoization
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 60000); // refresh every minute
    return () => clearInterval(id);
  }, []);

  // Real API hooks
  const { predictions, isLoading: isPredLoading } = useStockPrediction(
    selectedStock,
    horizonDays,
    { enabled: !!selectedStock }
  );

  const { sentiment: sentimentData, isLoading: isSentLoading } = useSentiment(selectedStock);
  const { recommendation: apiRecommendation } = useTradeRecommendation(selectedStock);
  const { volumePredictions } = useVolumePrediction(selectedStock, horizonDays);
  const { liquidityForecasts } = useLiquidityPrediction(selectedStock, horizonDays);

  const stock = useMemo(() =>
    stocks?.find(s => s.ticker === selectedStock),
    [stocks, selectedStock]
  );

  // Build chart data from predictions (real API data)
  const chartData = useMemo(() => {
    if (!stock) return [];

    const data = [];
    const currentPrice = parseFloat(stock.dernier) || 0;

    // Today's price as the anchor point
    data.push({
      date: new Date(now).toLocaleDateString('fr-FR', { month: 'short', day: 'numeric' }),
      price: currentPrice,
      type: 'historical'
    });

    // Prediction points from actual API
    if (predictions && predictions.length > 0) {
      predictions.forEach((pred, idx) => {
        const targetDate = pred.target_date
          ? new Date(pred.target_date).toLocaleDateString('fr-FR', { month: 'short', day: 'numeric' })
          : new Date(now + (idx + 1) * 86400000).toLocaleDateString('fr-FR', { month: 'short', day: 'numeric' });

        data.push({
          date: targetDate,
          prediction: pred.predicted_close ?? pred.predicted_price,
          upper: pred.confidence_upper ?? null,
          lower: pred.confidence_lower ?? null,
          type: 'prediction'
        });
      });
    }

    return data;
  }, [stock, predictions, now]);

  // Volume chart data from real predictions
  const volumeChartData = useMemo(() => {
    if (!volumePredictions || volumePredictions.length === 0) return [];
    return volumePredictions.map(v => ({
      date: new Date(v.target_date).toLocaleDateString('fr-FR', { month: 'short', day: 'numeric' }),
      volume: v.predicted_volume,
    }));
  }, [volumePredictions]);

  // Liquidity data from real predictions
  const liquidityData = useMemo(() => {
    if (!liquidityForecasts || liquidityForecasts.length === 0) return null;
    const latest = liquidityForecasts[0];
    return {
      tier: latest.predicted_tier,
      probHigh: (latest.prob_high * 100).toFixed(1),
      probMedium: (latest.prob_medium * 100).toFixed(1),
      probLow: (latest.prob_low * 100).toFixed(1),
    };
  }, [liquidityForecasts]);

  // Build recommendation from real API data
  const recommendation = useMemo(() => {
    if (!apiRecommendation) return null;

    const actionMap = {
      'buy': 'ACHETER', 'strong_buy': 'ACHETER',
      'sell': 'VENDRE', 'strong_sell': 'VENDRE',
      'hold': 'CONSERVER',
    };
    const rawAction = (apiRecommendation.action || 'hold').toLowerCase();
    const action = actionMap[rawAction] || 'CONSERVER';
    const confidence = apiRecommendation.confidence != null
      ? (apiRecommendation.confidence * 100).toFixed(0)
      : '–';
    const reasoning = apiRecommendation.reasoning || 'Aucune analyse disponible.';

    // Calculate predicted price change from predictions
    let priceChange = '–';
    if (predictions && predictions.length > 0 && stock) {
      const currentPrice = parseFloat(stock.dernier) || 0;
      const lastPred = predictions[predictions.length - 1];
      const predictedPrice = lastPred.predicted_close ?? lastPred.predicted_price ?? 0;
      if (currentPrice > 0 && predictedPrice > 0) {
        priceChange = (((predictedPrice - currentPrice) / currentPrice) * 100).toFixed(2);
      }
    }

    return { action, confidence, reason: reasoning, priceChange };
  }, [apiRecommendation, predictions, stock]);

  // Sentiment display
  const sentimentDisplay = useMemo(() => {
    if (!sentimentData) return null;
    const score = sentimentData.score != null ? (sentimentData.score * 100).toFixed(0) : null;
    return {
      score,
      label: sentimentData.sentiment || 'Neutre',
      articleCount: sentimentData.article_count || 0,
    };
  }, [sentimentData]);

  // Technical indicators from real stock data
  const technicalIndicators = useMemo(() => {
    if (!stock) return null;
    const price = parseFloat(stock.dernier) || 0;
    const open = parseFloat(stock.ouverture) || price;
    const high = parseFloat(stock.plus_haut) || price;
    const low = parseFloat(stock.plus_bas) || price;
    const volume = stock.volumeD || stock.volume || '–';

    return {
      open: open.toFixed(3),
      high: high.toFixed(3),
      low: low.toFixed(3),
      volume,
      support: low.toFixed(3),
      resistance: high.toFixed(3),
    };
  }, [stock]);

  if (isLoading) {
    return (
      <div className="p-4 md:p-6 lg:p-8">
        <div className="max-w-[1600px] mx-auto">
          <div className="animate-pulse space-y-6">
            <div className="h-12 bg-finance-bg rounded w-1/3"></div>
            <div className="h-64 bg-finance-bg rounded"></div>
            <div className="h-48 bg-finance-bg rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 md:p-6 lg:p-8">
      <div className="max-w-[1600px] mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl md:text-3xl font-bold text-finance-text-primary mb-2">
            Analyse d'une Valeur
          </h1>
          <p className="text-finance-text-secondary">
            Analyse détaillée avec prévisions et recommandations IA
          </p>
        </div>

        {/* Stock Selector */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-finance-text-primary mb-2">
            Sélectionner une valeur
          </label>
          <select
            value={selectedStock}
            onChange={(e) => setSelectedStock(e.target.value)}
            className="w-full md:w-96 px-4 py-3 bg-finance-card border border-finance-border rounded-lg text-finance-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            <option value="">-- Choisir une valeur --</option>
            {stocks?.map((s) => (
              <option key={s.ticker} value={s.ticker}>
                {s.ticker} - {s.societe}
              </option>
            ))}
          </select>
        </div>

        {!selectedStock ? (
          <div className="finance-card p-12 rounded-lg text-center">
            <BarChart3 className="w-16 h-16 text-finance-text-secondary mx-auto mb-4" />
            <p className="text-finance-text-secondary text-lg">
              Sélectionnez une valeur pour voir l'analyse détaillée
            </p>
          </div>
        ) : (
          <>
            {/* Stock Info Header */}
            <div className="finance-card p-6 rounded-lg mb-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-finance-text-primary">{stock?.ticker}</h2>
                  <p className="text-finance-text-secondary">{stock?.societe}</p>
                </div>
                <div className="text-right">
                  <p className="text-3xl font-bold text-finance-text-primary">
                    {parseFloat(stock?.dernier || 0).toFixed(3)} TND
                  </p>
                  <p className={`text-lg font-semibold ${parseFloat(stock?.variation || 0) >= 0 ? 'text-success-500' : 'text-danger-500'}`}>
                    {parseFloat(stock?.variation || 0) >= 0 ? '+' : ''}{parseFloat(stock?.variation || 0).toFixed(2)}%
                  </p>
                </div>
              </div>
            </div>

            {/* Price Chart with Predictions */}
            <div className="finance-card p-6 rounded-lg mb-6">
              <h3 className="text-lg font-semibold text-finance-text-primary mb-4">
                Prix Actuel + Prévisions {horizonDays} Jours
              </h3>
              {isPredLoading ? (
                <div className="h-80 flex items-center justify-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500"></div>
                </div>
              ) : (
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={chartData}>
                    <CartesianGrid {...chartConfig.cartesianGrid} />
                    <XAxis dataKey="date" {...chartConfig.axis} />
                    <YAxis domain={['auto', 'auto']} {...chartConfig.axis} />
                    <Tooltip {...chartConfig.tooltip} />
                    <Legend {...chartConfig.legend} />
                    <Line
                      {...lineConfigs.historical}
                      dataKey="price"
                      name="Prix Historique"
                    />
                    <Line
                      {...lineConfigs.prediction}
                      dataKey="prediction"
                      name="Prévision"
                      connectNulls
                    />
                    <Line
                      type="monotone"
                      dataKey="upper"
                      name="Intervalle Sup."
                      stroke={chartColors.success}
                      strokeWidth={1}
                      strokeDasharray="3 3"
                      dot={false}
                      connectNulls
                    />
                    <Line
                      type="monotone"
                      dataKey="lower"
                      name="Intervalle Inf."
                      stroke={chartColors.danger}
                      strokeWidth={1}
                      strokeDasharray="3 3"
                      dot={false}
                      connectNulls
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>

            {/* Sentiment Analysis */}
            <div className="finance-card p-6 rounded-lg mb-6">
              <h3 className="text-lg font-semibold text-finance-text-primary mb-4">
                Analyse de Sentiment NLP
              </h3>
              {isSentLoading ? (
                <div className="h-32 flex items-center justify-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-primary-500"></div>
                </div>
              ) : sentimentDisplay ? (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-finance-bg p-4 rounded text-center">
                    <p className="text-sm text-finance-text-secondary mb-1">Score</p>
                    <p className={`text-3xl font-bold ${
                      parseFloat(sentimentDisplay.score) >= 60 ? 'text-success-500' :
                      parseFloat(sentimentDisplay.score) <= 40 ? 'text-danger-500' :
                      'text-warning-500'
                    }`}>
                      {sentimentDisplay.score}%
                    </p>
                  </div>
                  <div className="bg-finance-bg p-4 rounded text-center">
                    <p className="text-sm text-finance-text-secondary mb-1">Classification</p>
                    <p className={`text-2xl font-bold ${
                      sentimentDisplay.label === 'positive' ? 'text-success-500' :
                      sentimentDisplay.label === 'negative' ? 'text-danger-500' :
                      'text-warning-500'
                    }`}>
                      {sentimentDisplay.label === 'positive' ? 'Positif' :
                       sentimentDisplay.label === 'negative' ? 'Négatif' : 'Neutre'}
                    </p>
                  </div>
                  <div className="bg-finance-bg p-4 rounded text-center">
                    <p className="text-sm text-finance-text-secondary mb-1">Articles Analysés</p>
                    <p className="text-2xl font-bold text-finance-text-primary">
                      {sentimentDisplay.articleCount}
                    </p>
                  </div>
                </div>
              ) : (
                <p className="text-center text-finance-text-secondary py-4">
                  Aucune donnée de sentiment disponible pour cette valeur.
                </p>
              )}
            </div>

            {/* Volume Predictions */}
            {volumeChartData.length > 0 && (
              <div className="finance-card p-6 rounded-lg mb-6">
                <h3 className="text-lg font-semibold text-finance-text-primary mb-4 flex items-center gap-2">
                  <Volume2 className="w-5 h-5" />
                  Prévisions de Volume ({horizonDays}j)
                </h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={volumeChartData}>
                    <CartesianGrid {...chartConfig.cartesianGrid} />
                    <XAxis dataKey="date" {...chartConfig.axis} />
                    <YAxis {...chartConfig.axis} />
                    <Tooltip {...chartConfig.tooltip} />
                    <Bar dataKey="volume" fill={chartColors.info} name="Volume Prévu" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Liquidity + Market Data */}
            <div className="finance-card p-6 rounded-lg mb-6">
              <h3 className="text-lg font-semibold text-finance-text-primary mb-4">
                Données de Marché
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {technicalIndicators && (
                  <>
                    <div className="bg-finance-bg p-4 rounded">
                      <p className="text-sm text-finance-text-secondary mb-1">Ouverture</p>
                      <p className="text-xl font-bold text-finance-text-primary">{technicalIndicators.open} TND</p>
                    </div>
                    <div className="bg-finance-bg p-4 rounded">
                      <p className="text-sm text-finance-text-secondary mb-1">Plus Haut</p>
                      <p className="text-xl font-bold text-success-500">{technicalIndicators.high} TND</p>
                    </div>
                    <div className="bg-finance-bg p-4 rounded">
                      <p className="text-sm text-finance-text-secondary mb-1">Plus Bas</p>
                      <p className="text-xl font-bold text-danger-500">{technicalIndicators.low} TND</p>
                    </div>
                    <div className="bg-finance-bg p-4 rounded">
                      <p className="text-sm text-finance-text-secondary mb-1">Volume du Jour</p>
                      <p className="text-xl font-bold text-finance-text-primary">{technicalIndicators.volume}</p>
                    </div>
                    <div className="bg-finance-bg p-4 rounded">
                      <p className="text-sm text-finance-text-secondary mb-1">Support (Bas Jour)</p>
                      <p className="text-xl font-bold text-success-500">{technicalIndicators.support} TND</p>
                    </div>
                    <div className="bg-finance-bg p-4 rounded">
                      <p className="text-sm text-finance-text-secondary mb-1">Résistance (Haut Jour)</p>
                      <p className="text-xl font-bold text-danger-500">{technicalIndicators.resistance} TND</p>
                    </div>
                  </>
                )}
                {liquidityData && (
                  <div className="bg-finance-bg p-4 rounded md:col-span-2 lg:col-span-3">
                    <p className="text-sm text-finance-text-secondary mb-2 flex items-center gap-1">
                      <Droplets className="w-4 h-4" /> Prévision de Liquidité
                    </p>
                    <div className="flex items-center gap-6">
                      <span className={`text-xl font-bold ${
                        liquidityData.tier === 'high' ? 'text-success-500' :
                        liquidityData.tier === 'medium' ? 'text-warning-500' :
                        'text-danger-500'
                      }`}>
                        {liquidityData.tier === 'high' ? 'Élevée' :
                         liquidityData.tier === 'medium' ? 'Moyenne' : 'Faible'}
                      </span>
                      <span className="text-sm text-finance-text-secondary">
                        Haute: {liquidityData.probHigh}% · Moyenne: {liquidityData.probMedium}% · Faible: {liquidityData.probLow}%
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* AI Recommendation */}
            {recommendation && (
              <div className={`finance-card p-6 rounded-lg border-2 ${
                recommendation.action === 'ACHETER' ? 'border-success-500' :
                recommendation.action === 'VENDRE' ? 'border-danger-500' :
                'border-warning-500'
              }`}>
                <h3 className="text-lg font-semibold text-finance-text-primary mb-4 flex items-center gap-2">
                  <Activity className="w-5 h-5" />
                  Recommandation de l'Agent IA
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="text-center">
                    <p className="text-sm text-finance-text-secondary mb-2">Action Recommandée</p>
                    <p className={`text-3xl font-bold ${
                      recommendation.action === 'ACHETER' ? 'text-success-500' :
                      recommendation.action === 'VENDRE' ? 'text-danger-500' :
                      'text-warning-500'
                    }`}>
                      {recommendation.action}
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-finance-text-secondary mb-2">Score de Confiance</p>
                    <div className="relative pt-1">
                      <div className="flex mb-2 items-center justify-center">
                        <div className="text-3xl font-bold text-primary-500">
                          {recommendation.confidence}%
                        </div>
                      </div>
                      <div className="overflow-hidden h-2 text-xs flex rounded bg-finance-bg">
                        <div
                          style={{ width: `${recommendation.confidence}%` }}
                          className={`shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center ${
                            parseFloat(recommendation.confidence) > 70 ? 'bg-success-500' :
                            parseFloat(recommendation.confidence) > 50 ? 'bg-warning-500' :
                            'bg-danger-500'
                          }`}
                        ></div>
                      </div>
                    </div>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-finance-text-secondary mb-2">Variation Prévue (5j)</p>
                    <p className={`text-3xl font-bold ${
                      parseFloat(recommendation.priceChange) >= 0 ? 'text-success-500' : 'text-danger-500'
                    }`}>
                      {parseFloat(recommendation.priceChange) >= 0 ? '+' : ''}{recommendation.priceChange}%
                    </p>
                  </div>
                </div>
                <div className="mt-6 p-4 bg-finance-bg rounded">
                  <p className="text-sm text-finance-text-secondary mb-1">Analyse:</p>
                  <p className="text-finance-text-primary">{recommendation.reason}</p>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default StockAnalysis;
