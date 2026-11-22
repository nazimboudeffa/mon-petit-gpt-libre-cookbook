# Cinquième partie --- Aller plus loin

## Chapitre 14 --- De notre mini-GPT à GPT-2

Notre modèle est un GPT miniature : il partage des principes
architecturaux avec des modèles plus grands, mais il ne possède ni leur
échelle, ni leurs données, ni leurs capacités.

Un modèle comme GPT-2 utilise notamment :

-   un tokenizer sous-mots ;
-   des embeddings de grande dimension ;
-   plusieurs couches Transformer ;
-   beaucoup plus de paramètres ;
-   un corpus d'entraînement bien plus vaste ;
-   un entraînement et une évaluation à une échelle nettement
    supérieure.

Il est important de distinguer deux expressions :

**Construire un GPT from scratch** signifie programmer soi-même
l'architecture et les mécanismes.

**Entraîner un GPT from scratch** signifie initialiser les paramètres
sans partir des poids d'un modèle pré-entraîné, puis apprendre ces
paramètres à partir de données.

On peut construire soi-même le code d'un modèle et utiliser des poids
pré-entraînés. Ce sont deux décisions différentes.
