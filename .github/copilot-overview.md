CAHIER DES CHARGES : 
 Système d'Assistant Intelligent de Trading pour la BVMT 
PROBLÉMATIQUE IHEC-CODELAB 2.0


1.1 Contexte
La Bourse des Valeurs Mobilières de Tunis (BVMT) s'inscrit dans une dynamique de modernisation et de digitalisation de ses services. Dans un environnement financier de plus en plus complexe et volatil, les investisseurs tunisiens ont besoin d'outils intelligents qui les accompagnent dans leurs décisions d'investissement tout en garantissant la sécurité et la conformité réglementaire.
Le marché tunisien présente des spécificités uniques :
    • Liquidité variable selon les valeurs
    • Sources d'information multilingues (français, arabe)
    • Besoin accru de surveillance des manipulations de marché ***
    • Cadre réglementaire strict (CMF - Conseil du Marché Financier)
1.2 Le Défi
Concevoir et développer un système intelligent intégré qui combine analyse prédictive, détection d'anomalies, analyse de sentiment et aide à la décision pour offrir aux investisseurs tunisiens un compagnon de trading complet et sécurisé.

2. Votre Mission
Les équipes participantes devront créer un prototype d'Assistant Intelligent de Trading capable de :
Fonctionnalités Core : 
A. Prévision des Prix et de la Liquidité (ML/Deep Learning)
    • Prédire les prix à court terme (1 à 5 jours) pour les principales valeurs de la BVMT
    • Anticiper les périodes de faible/forte liquidité
    • Identifier les meilleurs moments pour entrer/sortir d'une position
B. Analyse de Sentiment de Marché (NLP)
    • Collecter et analyser automatiquement les actualités financières tunisiennes
    • Classifier le sentiment (positif/négatif/neutre) pour chaque valeur cotée
    • Corréler le sentiment médiatique avec les mouvements de prix *** (optional) 
C. Détection d'Anomalies (Surveillance de Marché) *** 
    • Identifier en temps réel les comportements suspects (pics de volume, variations anormales)
    • Générer des alertes pour protéger l'investisseur
    • Détecter les potentielles manipulations de marché
D. Agent de Décision Augmentée (IA + Interface)
    • Recommander des actions concrètes (acheter/vendre/conserver)
    • Simuler l'impact de décisions sur un portefeuille virtuel
    • Expliquer chaque recommandation de manière transparente
    • Suivre et optimiser un portefeuille multi-actifs

3. Spécifications Techniques Détaillées
3.1 Module 1 : Prévision des Prix et de la Liquidité
Objectifs Techniques
    • Prédire le prix de clôture des 5 prochains jours ouvrables
    • Estimer le volume de transactions journalier
    • Calculer la probabilité de forte/faible liquidité
Technologies Requises
    • Time Series Forecasting : 
    • Feature Engineering : 
    • Frameworks : 
Données Fournies
    • Historique 3 ans minimum : Date, Open, High, Low, Close, Volume pour 10-15 valeurs BVMT
    • Indices : TUNINDEX, TUNINDEX20
Livrables Attendus
    • Modèle entraîné avec métriques (RMSE, MAE, Directional Accuracy)
    • Visualisation : Graphiques prévisions vs réel avec intervalles de confiance
    • API ou fonction Python retournant prévisions pour une valeur donnée
    • Pipeline de prévision temps réel (bonus significatif)

4.2 Module 2 : Analyse de Sentiment
Objectifs Techniques
    • Scraper automatiquement 3+ sources d'actualités tunisiennes
    • Classifier sentiment pour chaque article mentionnant une valeur BVMT
    • Calculer un "Score de Sentiment Quotidien" agrégé par entreprise
Technologies Requises
    • Web Scraping 
    • NLP 
    • Multilinguisme : Gestion français + arabe 

4.3 Module 3 : Détection d'Anomalies
Objectifs Techniques
    • Détecter en temps réel (ou quasi-réel) :
        ◦ Pics de volume (>3 écarts-types de la moyenne)
        ◦ Variations de prix anormales (>5% en 1 heure sans news)
        ◦ Patterns suspects (séquences d'ordres inhabituelles)
Technologies Requises
    • Anomaly Detection 
    • Streaming (bonus) 
    • Alerting : Système de notification 
Données Fournies
    • Historique tick-by-tick ou 1min pour 5 valeurs (1 mois)
    • Liste d'anomalies connues (dates + description) pour validation
Livrables Attendus
    • Modèle de détection avec Precision/Recall/F1-Score
    • Interface visuelle montrant :
        ◦ Graphique temps réel avec zones d'alerte
        ◦ Top 5 anomalies détectées aujourd'hui
    • Système d'alerte fonctionnel (pop-up ou notification)

4.4 Module 4 : Agent de Décision et Gestion de Portefeuille
Objectifs Techniques
    • Profil Utilisateur :(conservateur/modéré/agressif) 
    • Agrégation Intelligente :
On peut combiner les autres parties ( devenir des moyen d’aide à la decision)
    • Simulation de Portefeuille :
        ◦ Capital initial virtuel (ex: 10,000 TND)
        ◦ Tracking performance (gains/pertes)
        ◦ Calcul de métriques : ROI, Sharpe Ratio, Max Drawdown
    • ( Explainability : Chaque recommandation doit être justifiée en langage naturel *****)
Technologies Requises
    • Reinforcement Learning pour optimisation portefeuille
    • Rule-Based System (minimum viable) : Logique if/else sophistiquée
Livrables Attendus
    • Interface avec :
        ◦ Vue portefeuille actuel (composition, valeur totale)
        ◦ Recommandations du jour pour 5-10 valeurs
        ◦ Bouton "Expliquer" pour chaque recommandation
        ◦ Graphique performance historique du portefeuille


5. Interface Utilisateur (Dashboard)
5.1 Pages Obligatoires
Page 1 : Vue d'Ensemble du Marché
    • Indices principaux (TUNINDEX) avec variation du jour
    • Top 5 gagnants/perdants
    • Presenter sentiment global du marché 
    • Alertes récentes (anomalies détectées)
Page 2 : Analyse d'une Valeur Spécifique
    • Sélecteur de valeur (dropdown)
    • Graphique prix historique + prévisions 5 jours
    • Timeline sentiment (graphique ligne)
    • Indicateurs techniques (RSI, MACD)
    • Recommandation de l'agent : ACHETER / VENDRE / CONSERVER avec score de confiance
Page 3 : Mon Portefeuille
    • Liste des positions actuelles (valeur, quantité, prix d'achat, P&L)
    • Graphique répartition (pie chart)
    • Performance globale (ROI, graphique évolution capital)
    • Suggestions d'optimisation
Page 4 : Surveillance & Alertes
    • Feed en temps réel des anomalies détectées
    • Filtres par type (volume, prix, news)
    • Historique des alertes avec actions prises

—


8. Livrables Finaux (à remettre en fin de hackathon)
8.1 Livrables Techniques Obligatoires
    1. Code Source Complet
        ◦ Repository Git (GitHub/GitLab) ou archive ZIP
        ◦ README.md avec instructions d'installation
        ◦ Requirements.txt (Python) ou package.json (Node)
    2. Application Fonctionnelle
        ◦ Déployée localement OU Hébergée en ligne (Vercel, Render, Heroku gratuit)
        ◦ URL d'accès + identifiants de démo
    3. Documentation Technique
        ◦ Architecture du système (diagramme)
        ◦ Choix des modèles ML/DL justifiés
        ◦ Métriques de performance (accuracy, RMSE, F1-score, etc.)
        ◦ Limites identifiées et améliorations futures
    4. Notebooks Jupyter (recommandé)
        ◦ Analyse exploratoire des données
        ◦ Entraînement des modèles avec résultats
        ◦ Visualisations clés
8.2 Livrables de Présentation
    5. Pitch Deck (10-15 min maximum)
    6. Vidéo Démo (3-5 minutes)
        ◦ Parcours utilisateur complet
        ◦ Cas d'usage concret : "Je veux investir 5000 TND, que me recommande le système ?"

11. Scénarios d'Usage Attendus (User Stories)
Scénario 1 : L'Investisseur Débutant
Persona : Ahmed, 28 ans, ingénieur, veut investir ses économies (5000 TND) mais ne connaît rien à la bourse.
Parcours :
    1. Ahmed ouvre l'application et répond au questionnaire de profil → Profil "Modéré" détecté
    2. Le système lui recommande un portefeuille diversifié : 40% actions stables, 30% obligations, 30% liquidité
    3. Ahmed clique sur "Tunisie Télécom" → Prévision : +2.5% dans 5 jours, Sentiment : Positif, Recommandation : ACHETER
    4. Ahmed demande "Pourquoi ?" → Chatbot explique : "Actualités récentes positives (nouveau contrat) + tendance haussière prévue + aucune anomalie détectée"
    5. Ahmed achète 100 actions → Portefeuille mis à jour en temps réel
Scénario 2 : Le Trader Averti
Persona : Leila, 35 ans, trader active, cherche à optimiser sa stratégie.
Parcours :
    1. Leila consulte le dashboard → Alerte : "Volume anormal détecté sur SFBT (+800%)"
    2. Elle clique sur l'alerte → Graphique montre pic de 10h23, article news apparu à 10h20 ("SFBT annonce partenariat majeur")
    3. Module sentiment confirme : score +0.85 / explication texte
    4. Mais module prévision indique : "Volatilité élevée prévue, prudence recommandée"
    5. Leila décide d'attendre 24h → Le lendemain, prix stabilisé, elle entre en position
    6. Système trackera sa performance et apprendra de cette décision
Scénario 3 : Le Régulateur (CMF)
Persona : Inspecteur CMF, utilise le système pour surveillance marché.
Parcours :
    1. Reçoit alerte : "Anomalie détectée : Délice Holding, variation +12% sans news significative"
    2. Consulte dashboard : Volume 10x supérieur à moyenne, aucun article positif récent (sentiment neutre)
    3. Suspect de manipulation → Déclenche enquête interne
    4. Système lui fournit timeline détaillée : heures précises, ordres suspects
