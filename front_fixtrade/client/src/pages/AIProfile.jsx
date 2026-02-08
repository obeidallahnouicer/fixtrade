import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { getAIStatus, submitProfileQuestionnaire, getAIRecommendations } from '../services/api';
import { usePortfolioSnapshot, usePortfolioPerformance } from '../hooks/usePortfolio';
import { getStoredPortfolioId } from '../hooks/usePortfolio';
import { Brain, Shield, TrendingUp, Zap, CheckCircle, AlertTriangle, Activity, Target, BarChart3, Briefcase } from 'lucide-react';

const RISK_QUESTIONS = [
  {
    id: 'horizon',
    question: 'Quel est votre horizon d\'investissement ?',
    options: [
      { value: 'court', label: 'Court terme (< 1 an)', score: 1 },
      { value: 'moyen', label: 'Moyen terme (1-5 ans)', score: 2 },
      { value: 'long', label: 'Long terme (> 5 ans)', score: 3 },
    ],
  },
  {
    id: 'loss_tolerance',
    question: 'Quelle perte maximale acceptez-vous sur votre portefeuille ?',
    options: [
      { value: '5', label: 'Jusqu\'à 5%', score: 1 },
      { value: '15', label: 'Jusqu\'à 15%', score: 2 },
      { value: '30', label: 'Jusqu\'à 30%', score: 3 },
    ],
  },
  {
    id: 'experience',
    question: 'Quelle est votre expérience en investissement ?',
    options: [
      { value: 'debutant', label: 'Débutant', score: 1 },
      { value: 'intermediaire', label: 'Intermédiaire', score: 2 },
      { value: 'expert', label: 'Expert', score: 3 },
    ],
  },
  {
    id: 'objective',
    question: 'Quel est votre objectif principal ?',
    options: [
      { value: 'preservation', label: 'Préservation du capital', score: 1 },
      { value: 'revenus', label: 'Revenus réguliers (dividendes)', score: 2 },
      { value: 'croissance', label: 'Croissance maximale', score: 3 },
    ],
  },
  {
    id: 'reaction',
    question: 'En cas de chute de 20% du marché, que faites-vous ?',
    options: [
      { value: 'vendre', label: 'Je vends immédiatement', score: 1 },
      { value: 'attendre', label: 'J\'attends que ça remonte', score: 2 },
      { value: 'acheter', label: 'J\'achète plus (opportunité)', score: 3 },
    ],
  },
];

const RISK_PROFILES = {
  conservateur: { label: 'Conservateur', color: 'text-blue-400', bg: 'bg-blue-500/10', icon: Shield, description: 'Priorité à la préservation du capital avec un risque minimal.' },
  modere: { label: 'Modéré', color: 'text-yellow-400', bg: 'bg-yellow-500/10', icon: Target, description: 'Équilibre entre croissance et sécurité.' },
  agressif: { label: 'Agressif', color: 'text-red-400', bg: 'bg-red-500/10', icon: Zap, description: 'Recherche de rendements élevés avec tolérance au risque.' },
};

const AIProfile = () => {
  const portfolioId = getStoredPortfolioId();
  const [answers, setAnswers] = useState({});
  const [profileResult, setProfileResult] = useState(null);
  const [currentStep, setCurrentStep] = useState(0);

  // AI status
  const { data: aiStatus, isLoading: statusLoading } = useQuery({
    queryKey: ['ai-status'],
    queryFn: async () => {
      const res = await getAIStatus();
      return res.data;
    },
    retry: 1,
  });

  // AI recommendations
  const { data: recommendations } = useQuery({
    queryKey: ['ai-recommendations-overview'],
    queryFn: async () => {
      const res = await getAIRecommendations(5);
      return res.data;
    },
    retry: 1,
  });

  // Portfolio data
  const { data: snapshot } = usePortfolioSnapshot(portfolioId);
  const { data: performance } = usePortfolioPerformance(portfolioId);

  // Submit questionnaire
  const profileMutation = useMutation({
    mutationFn: async (formAnswers) => {
      const res = await submitProfileQuestionnaire(formAnswers);
      return res.data;
    },
    onSuccess: (data) => {
      setProfileResult(data);
    },
  });

  const handleAnswer = (questionId, value, score) => {
    setAnswers((prev) => ({ ...prev, [questionId]: { value, score } }));
    if (currentStep < RISK_QUESTIONS.length - 1) {
      setCurrentStep((s) => s + 1);
    }
  };

  const handleSubmitProfile = () => {
    const payload = {};
    Object.entries(answers).forEach(([key, { value }]) => {
      payload[key] = value;
    });
    profileMutation.mutate(payload);
  };

  const totalScore = Object.values(answers).reduce((acc, { score }) => acc + score, 0);
  const maxScore = RISK_QUESTIONS.length * 3;
  const allAnswered = Object.keys(answers).length === RISK_QUESTIONS.length;

  const computedProfile = totalScore <= 7 ? 'conservateur' : totalScore <= 11 ? 'modere' : 'agressif';

  return (
    <div className="p-4 md:p-6 lg:p-8">
      <div className="max-w-[1400px] mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl md:text-3xl font-bold text-finance-text-primary mb-2 flex items-center gap-3">
            <Brain className="w-8 h-8 text-primary-400" />
            Agent IA & Profil Investisseur
          </h1>
          <p className="text-finance-text-secondary">
            Configurez votre profil de risque et consultez les recommandations de l'intelligence artificielle
          </p>
        </div>

        {/* AI Status Card */}
        <div className="mb-6">
          <div className="finance-card p-6 rounded-lg">
            <h2 className="text-lg font-semibold text-finance-text-primary mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-primary-400" />
              Statut de l'Agent IA
            </h2>
            {statusLoading ? (
              <div className="animate-pulse flex gap-4">
                <div className="h-16 bg-finance-bg rounded flex-1"></div>
                <div className="h-16 bg-finance-bg rounded flex-1"></div>
                <div className="h-16 bg-finance-bg rounded flex-1"></div>
              </div>
            ) : aiStatus ? (
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-finance-bg rounded-lg p-4 text-center">
                  <div className={`w-3 h-3 rounded-full mx-auto mb-2 ${aiStatus.status === 'active' || aiStatus.status === 'ready' ? 'bg-success-500' : 'bg-yellow-500'}`}></div>
                  <p className="text-sm text-finance-text-secondary">Statut</p>
                  <p className="font-semibold text-finance-text-primary capitalize">{aiStatus.status || 'Prêt'}</p>
                </div>
                <div className="bg-finance-bg rounded-lg p-4 text-center">
                  <Brain className="w-5 h-5 text-primary-400 mx-auto mb-2" />
                  <p className="text-sm text-finance-text-secondary">Modèles</p>
                  <p className="font-semibold text-finance-text-primary">
                    {aiStatus.models_loaded || aiStatus.modules?.length || 'Ensemble'}
                  </p>
                </div>
                <div className="bg-finance-bg rounded-lg p-4 text-center">
                  <Zap className="w-5 h-5 text-yellow-400 mx-auto mb-2" />
                  <p className="text-sm text-finance-text-secondary">Prédictions</p>
                  <p className="font-semibold text-finance-text-primary">
                    {aiStatus.predictions_today ?? aiStatus.total_predictions ?? 'Actives'}
                  </p>
                </div>
                <div className="bg-finance-bg rounded-lg p-4 text-center">
                  <Shield className="w-5 h-5 text-blue-400 mx-auto mb-2" />
                  <p className="text-sm text-finance-text-secondary">Version</p>
                  <p className="font-semibold text-finance-text-primary">{aiStatus.version || '1.0'}</p>
                </div>
              </div>
            ) : (
              <div className="text-center text-finance-text-secondary py-4">
                <AlertTriangle className="w-8 h-8 mx-auto mb-2 text-yellow-500" />
                <p>Impossible de contacter l'agent IA. Vérifiez que le backend est en cours d'exécution.</p>
              </div>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Risk Profile Questionnaire */}
          <div className="finance-card p-6 rounded-lg">
            <h2 className="text-lg font-semibold text-finance-text-primary mb-4 flex items-center gap-2">
              <Shield className="w-5 h-5 text-blue-400" />
              Questionnaire Profil de Risque
            </h2>

            {profileResult ? (
              <div className="text-center py-6">
                <CheckCircle className="w-12 h-12 text-success-500 mx-auto mb-3" />
                <h3 className="text-xl font-bold text-finance-text-primary mb-2">Profil Déterminé</h3>
                {(() => {
                  const pKey = profileResult.risk_profile || computedProfile;
                  const pData = RISK_PROFILES[pKey] || RISK_PROFILES.modere;
                  const PIcon = pData.icon;
                  return (
                    <div className={`${pData.bg} rounded-lg p-6 mt-4`}>
                      <PIcon className={`w-10 h-10 ${pData.color} mx-auto mb-2`} />
                      <p className={`text-2xl font-bold ${pData.color}`}>{pData.label}</p>
                      <p className="text-sm text-finance-text-secondary mt-2">{pData.description}</p>
                      {profileResult.max_loss_tolerance && (
                        <p className="text-sm text-finance-text-secondary mt-2">
                          Tolérance perte max: <span className="font-semibold text-finance-text-primary">{(profileResult.max_loss_tolerance * 100).toFixed(0)}%</span>
                        </p>
                      )}
                    </div>
                  );
                })()}
                <button
                  onClick={() => { setProfileResult(null); setAnswers({}); setCurrentStep(0); }}
                  className="mt-4 px-4 py-2 text-sm text-primary-400 hover:text-primary-300 transition-colors"
                >
                  Refaire le questionnaire
                </button>
              </div>
            ) : (
              <div>
                {/* Progress */}
                <div className="flex gap-1 mb-6">
                  {RISK_QUESTIONS.map((_, i) => (
                    <div
                      key={i}
                      className={`h-1.5 flex-1 rounded-full transition-colors ${
                        answers[RISK_QUESTIONS[i].id] ? 'bg-primary-500' : i === currentStep ? 'bg-primary-500/40' : 'bg-finance-bg'
                      }`}
                    />
                  ))}
                </div>

                {/* Current Question */}
                <div className="mb-6">
                  <p className="text-xs text-finance-text-secondary mb-1">
                    Question {currentStep + 1} / {RISK_QUESTIONS.length}
                  </p>
                  <p className="text-finance-text-primary font-medium mb-4">
                    {RISK_QUESTIONS[currentStep].question}
                  </p>
                  <div className="space-y-2">
                    {RISK_QUESTIONS[currentStep].options.map((opt) => (
                      <button
                        key={opt.value}
                        onClick={() => handleAnswer(RISK_QUESTIONS[currentStep].id, opt.value, opt.score)}
                        className={`w-full text-left p-3 rounded-lg border transition-colors ${
                          answers[RISK_QUESTIONS[currentStep].id]?.value === opt.value
                            ? 'border-primary-500 bg-primary-500/10 text-primary-400'
                            : 'border-finance-border text-finance-text-secondary hover:border-finance-text-secondary hover:bg-finance-bg'
                        }`}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Navigation */}
                <div className="flex justify-between items-center">
                  <button
                    onClick={() => setCurrentStep((s) => Math.max(0, s - 1))}
                    disabled={currentStep === 0}
                    className="px-4 py-2 text-sm text-finance-text-secondary hover:text-finance-text-primary disabled:opacity-30 transition-colors"
                  >
                    Précédent
                  </button>

                  {allAnswered && (
                    <button
                      onClick={handleSubmitProfile}
                      disabled={profileMutation.isPending}
                      className="px-6 py-2.5 bg-primary-500 text-white rounded-lg font-medium hover:bg-primary-600 transition-colors disabled:opacity-50"
                    >
                      {profileMutation.isPending ? 'Analyse...' : 'Déterminer mon profil'}
                    </button>
                  )}

                  <button
                    onClick={() => setCurrentStep((s) => Math.min(RISK_QUESTIONS.length - 1, s + 1))}
                    disabled={currentStep === RISK_QUESTIONS.length - 1}
                    className="px-4 py-2 text-sm text-finance-text-secondary hover:text-finance-text-primary disabled:opacity-30 transition-colors"
                  >
                    Suivant
                  </button>
                </div>

                {/* Score preview */}
                {Object.keys(answers).length > 0 && (
                  <div className="mt-4 pt-4 border-t border-finance-border">
                    <div className="flex justify-between text-xs text-finance-text-secondary">
                      <span>Score actuel: {totalScore} / {maxScore}</span>
                      <span className={RISK_PROFILES[computedProfile].color}>
                        Tendance: {RISK_PROFILES[computedProfile].label}
                      </span>
                    </div>
                    <div className="mt-1 h-2 bg-finance-bg rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-blue-500 via-yellow-500 to-red-500 rounded-full transition-all"
                        style={{ width: `${(totalScore / maxScore) * 100}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Portfolio Summary (if exists) */}
          <div className="finance-card p-6 rounded-lg">
            <h2 className="text-lg font-semibold text-finance-text-primary mb-4 flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-success-500" />
              Résumé Portefeuille IA
            </h2>
            {portfolioId && snapshot ? (
              <div>
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="bg-finance-bg rounded-lg p-4">
                    <p className="text-xs text-finance-text-secondary">Valeur Totale</p>
                    <p className="text-xl font-bold text-finance-text-primary">
                      {(snapshot.total_value || snapshot.portfolio?.total_value || 0).toLocaleString('fr-FR')} TND
                    </p>
                  </div>
                  <div className="bg-finance-bg rounded-lg p-4">
                    <p className="text-xs text-finance-text-secondary">Positions</p>
                    <p className="text-xl font-bold text-finance-text-primary">
                      {snapshot.positions?.length || snapshot.portfolio?.positions?.length || 0}
                    </p>
                  </div>
                </div>

                {performance && (
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-finance-bg rounded p-3">
                      <p className="text-xs text-finance-text-secondary">ROI</p>
                      <p className={`font-semibold ${(performance.roi || 0) >= 0 ? 'text-success-500' : 'text-danger-500'}`}>
                        {((performance.roi || 0) * 100).toFixed(2)}%
                      </p>
                    </div>
                    <div className="bg-finance-bg rounded p-3">
                      <p className="text-xs text-finance-text-secondary">Sharpe</p>
                      <p className="font-semibold text-finance-text-primary">
                        {(performance.sharpe_ratio || 0).toFixed(2)}
                      </p>
                    </div>
                    <div className="bg-finance-bg rounded p-3">
                      <p className="text-xs text-finance-text-secondary">Volatilité</p>
                      <p className="font-semibold text-finance-text-primary">
                        {((performance.volatility || 0) * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="bg-finance-bg rounded p-3">
                      <p className="text-xs text-finance-text-secondary">Win Rate</p>
                      <p className="font-semibold text-finance-text-primary">
                        {((performance.win_rate || 0) * 100).toFixed(0)}%
                      </p>
                    </div>
                  </div>
                )}

                <a
                  href="/portfolio"
                  className="mt-4 block text-center px-4 py-2 bg-primary-500/10 text-primary-400 rounded-lg text-sm hover:bg-primary-500/20 transition-colors"
                >
                  Voir le portefeuille complet →
                </a>
              </div>
            ) : (
              <div className="text-center py-8">
                <Briefcase className="w-10 h-10 text-finance-text-secondary mx-auto mb-3 opacity-50" />
                <p className="text-finance-text-secondary mb-3">Aucun portefeuille IA créé</p>
                <a
                  href="/portfolio"
                  className="inline-block px-6 py-2.5 bg-primary-500 text-white rounded-lg font-medium hover:bg-primary-600 transition-colors"
                >
                  Créer mon portefeuille
                </a>
              </div>
            )}
          </div>
        </div>

        {/* AI Recommendations */}
        <div className="finance-card p-6 rounded-lg mb-6">
          <h2 className="text-lg font-semibold text-finance-text-primary mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-success-500" />
            Recommandations IA du Jour
          </h2>
          {recommendations && recommendations.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {recommendations.map((rec, idx) => {
                const actionColor =
                  rec.action === 'BUY' || rec.action === 'buy' ? 'text-success-500 bg-success-500/10' :
                  rec.action === 'SELL' || rec.action === 'sell' ? 'text-danger-500 bg-danger-500/10' :
                  'text-yellow-400 bg-yellow-500/10';
                const actionLabel =
                  rec.action === 'BUY' || rec.action === 'buy' ? 'ACHETER' :
                  rec.action === 'SELL' || rec.action === 'sell' ? 'VENDRE' : 'CONSERVER';
                return (
                  <div key={idx} className="bg-finance-bg rounded-lg p-4">
                    <div className="flex justify-between items-start mb-2">
                      <p className="font-bold text-finance-text-primary">{rec.symbol}</p>
                      <span className={`text-xs font-semibold px-2 py-1 rounded ${actionColor}`}>
                        {actionLabel}
                      </span>
                    </div>
                    <p className="text-sm text-finance-text-secondary mb-2">{rec.reasoning || rec.reason || ''}</p>
                    <div className="flex justify-between text-xs text-finance-text-secondary">
                      <span>Confiance: <span className="text-finance-text-primary font-medium">{((rec.confidence || 0) * 100).toFixed(0)}%</span></span>
                      {rec.predicted_change != null && (
                        <span>
                          Δ Prévue: <span className={`font-medium ${rec.predicted_change >= 0 ? 'text-success-500' : 'text-danger-500'}`}>
                            {rec.predicted_change >= 0 ? '+' : ''}{(rec.predicted_change * 100).toFixed(1)}%
                          </span>
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-center text-finance-text-secondary py-6">
              Aucune recommandation disponible pour le moment. Les recommandations sont générées lorsque le marché est ouvert.
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default AIProfile;
