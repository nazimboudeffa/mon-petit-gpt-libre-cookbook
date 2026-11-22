# Deuxième partie --- Les briques du Transformer

## Chapitre 4 --- Les embeddings : donner une représentation aux tokens

Un identifiant de token est un entier arbitraire. Un réseau neuronal a
besoin de vecteurs continus qu'il peut ajuster pendant l'apprentissage.

Une couche d'embedding est une table de paramètres. Pour chaque
identifiant, elle renvoie un vecteur de taille `d_model`.

``` python
import torch
from torch import nn

vocab_size = 30
d_model = 16

embedding = nn.Embedding(vocab_size, d_model)
ids = torch.tensor([2, 5, 7, 2])

vecteurs = embedding(ids)
print(vecteurs.shape)  # (4, 16)
```

Les deux occurrences de l'identifiant 2 renvoient le même vecteur, car
elles consultent la même ligne de la table.

### 4.1 Ajouter l'information de position

L'attention seule ne sait pas naturellement si un token est le premier
ou le dixième de la séquence. Il faut donc injecter une information de
position.

Une version pédagogique utilise des embeddings de position appris :

``` python
block_size = 64
position_embedding = nn.Embedding(block_size, d_model)

positions = torch.arange(10)
pos = position_embedding(positions)
print(pos.shape)  # (10, 16)
```

On additionne ensuite les embeddings des tokens et ceux des positions :

\[ h_i = E(x_i) + P(i) \]

Cette somme donne au modèle à la fois une représentation du token et une
information sur son emplacement.

**Expérience**

Affichez la norme des vecteurs d'embedding avant et après quelques
étapes d'entraînement. Les vecteurs sont des paramètres appris ; ils
évoluent avec le reste du réseau.

------------------------------------------------------------------------
