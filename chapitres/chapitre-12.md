## Chapitre 12 --- Ce que ce mini-GPT ne sait pas faire

Un modèle entraîné sur un petit corpus n'est pas un assistant
généraliste.

Il ne dispose pas nécessairement :

-   d'une connaissance fiable du monde ;
-   d'une capacité de raisonnement robuste ;
-   d'une mémoire durable entre les sessions ;
-   d'une compréhension vérifiée des faits ;
-   d'une capacité à suivre des instructions complexes.

Il peut produire une phrase grammaticalement plausible mais
factuellement fausse. Il peut également reproduire des passages de son
corpus.

La génération de texte est une opération probabiliste conditionnée par
les paramètres et le contexte. Une sortie fluide n'est pas une preuve de
vérité.

### 12.1 Mémorisation et généralisation

La mémorisation consiste à reproduire des exemples ou des motifs vus
pendant l'entraînement. La généralisation consiste à obtenir de bons
résultats sur des exemples nouveaux issus d'une distribution pertinente.

Pour évaluer la généralisation, il faut un jeu de test indépendant, des
tâches représentatives et des critères définis à l'avance. Quelques
prompts choisis après coup ne constituent pas une évaluation suffisante.

### 12.2 La loss ne mesure pas tout

Une baisse de la cross-entropy est utile pour suivre l'apprentissage du
prochain token. Mais elle ne mesure pas directement :

-   la véracité ;
-   la sécurité ;
-   la pertinence d'une réponse ;
-   la capacité à utiliser des outils ;
-   la qualité d'un raisonnement complexe.

Il faut des évaluations complémentaires adaptées à l'usage visé.

------------------------------------------------------------------------
