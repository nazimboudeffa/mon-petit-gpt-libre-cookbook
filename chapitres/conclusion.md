# Conclusion --- Comprendre avant d'agrandir

Nous avons suivi le chemin d'un texte jusqu'à sa génération :

1.  le tokenizer transforme le texte en identifiants ;
2.  les embeddings transforment les identifiants en vecteurs ;
3.  l'attention permet aux positions de combiner des informations du
    contexte ;
4.  les blocs Transformer transforment ces représentations ;
5.  la tête de sortie produit des scores sur le vocabulaire ;
6.  la loss mesure l'écart avec le token attendu ;
7.  la rétropropagation et l'optimiseur ajustent les paramètres ;
8.  la génération répète la prédiction du prochain token.

Le modèle obtenu est volontairement modeste. Sa valeur est pédagogique :
chaque étape peut être inspectée, expliquée et modifiée.

La bonne suite n'est pas nécessairement de multiplier immédiatement les
paramètres. C'est de mener des expériences contrôlées, de comprendre les
résultats, puis d'introduire une complexité supplémentaire lorsque l'on
sait pourquoi elle est nécessaire.

**Un GPT devient moins mystérieux dès lors que l'on peut suivre chaque
nombre, chaque tenseur et chaque mise à jour de ses paramètres.**

------------------------------------------------------------------------
