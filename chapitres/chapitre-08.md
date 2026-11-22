## Chapitre 8 --- Comprendre la fonction de perte

Le modèle renvoie des logits : des scores non normalisés pour chaque
token possible. La fonction softmax les transforme en probabilités.

Pour une position donnée, si le prochain token attendu est `r`,
l'objectif est d'augmenter la probabilité attribuée à `r`.

La cross-entropy pénalise les probabilités faibles données à la bonne
réponse. Pour une cible (y) et une distribution (p) :

\[ L = -`\log `{=tex}p(y) \]

Une perte basse indique que le modèle attribue une forte probabilité aux
tokens attendus sur les exemples évalués. Elle ne garantit pas que le
texte généré soit vrai, utile ou grammatical.

### 8.1 Calculer une perte à la main

``` python
import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 0.5, -1.0]])
cible = torch.tensor([0])

loss = F.cross_entropy(logits, cible)
print(loss.item())
```

### 8.2 Pourquoi la perte est calculée sur toutes les positions

Dans notre modèle, chaque position produit une distribution sur le
vocabulaire. Nous comparons chaque distribution à son prochain token
attendu. La fonction `cross_entropy` reçoit donc les logits aplatis et
les cibles aplaties.

C'est ce que fait la méthode `forward` du modèle.

------------------------------------------------------------------------
