# Quatrième partie --- Comprendre ce que nous avons construit

## Chapitre 11 --- Expérimenter plutôt que simplement exécuter

Un bon apprentissage consiste à modifier une seule chose à la fois et à
observer son effet.

### Expérience A --- Supprimer le masque causal

Dans `MultiHeadCausalAttention`, retirez temporairement le
`masked_fill`. En entraînement, chaque position peut alors voir les
tokens futurs. La perte peut devenir artificiellement faible, mais le
modèle ne respecte plus la contrainte de génération autorégressive.

Réactivez ensuite le masque et comparez.

### Expérience B --- Réduire la taille du contexte

Essayez `block_size=8`, puis `block_size=32`, puis `block_size=64`. Un
contexte plus long donne accès à davantage de tokens précédents, mais
augmente le coût du calcul d'attention.

### Expérience C --- Modifier la dimension du modèle

Comparez `d_model=32`, `64` et `128`. Notez le nombre de paramètres, le
temps d'entraînement et la loss de validation. Ne concluez pas qu'un
modèle est meilleur sur la seule base de sa taille.

### Expérience D --- Réduire le corpus

Entraînez le modèle sur une seule phrase répétée. Observez comme il peut
rapidement reproduire les motifs du corpus sans acquérir une capacité
générale à écrire.

### Expérience E --- Changer le corpus

Utilisez un texte différent, gardez la même architecture et recommencez.
Le modèle apprend des régularités propres aux données qui lui sont
fournies.

------------------------------------------------------------------------
