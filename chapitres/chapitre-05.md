## Chapitre 5 --- L'attention : choisir quelles informations regarder

L'attention est l'un des mécanismes essentiels du Transformer. Pour
produire une représentation à une position donnée, le modèle peut
combiner les informations provenant de plusieurs positions du contexte.

L'intuition est la suivante : chaque position formule une requête
(Query), compare cette requête aux clés (Keys) des autres positions,
puis combine leurs valeurs (Values).

### 5.1 Query, Key et Value

À partir d'une représentation d'entrée (X), trois projections apprises
produisent :

\[ Q = XW_Q,`\quad `{=tex}K = XW_K,`\quad `{=tex}V = XW_V \]

-   Query : ce que la position cherche.
-   Key : ce que chaque position propose comme information repérable.
-   Value : l'information effectivement transmise.

Les scores de compatibilité sont calculés par produit scalaire :

\[ S = `\frac{QK^\top}{\sqrt{d_k}}`{=tex} \]

La division par (`\sqrt{d_k}`{=tex}) évite que les scores deviennent
trop grands lorsque la dimension des clés augmente.

On applique ensuite softmax sur chaque ligne :

\[ A = `\operatorname{softmax}`{=tex}(S) \]

Enfin, les valeurs sont combinées :

\[ `\operatorname{Attention}`{=tex}(Q,K,V) = AV \]

### 5.2 Un calcul jouet avec PyTorch

``` python
import math
import torch
import torch.nn.functional as F

Q = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0]])

K = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0]])

V = torch.tensor([[10.0, 0.0],
                  [0.0, 20.0]])

scores = Q @ K.T / math.sqrt(Q.size(-1))
poids = F.softmax(scores, dim=-1)
sortie = poids @ V

print("Scores :\n", scores)
print("Poids :\n", poids)
print("Sortie :\n", sortie)
```

Les poids d'attention indiquent dans quelle proportion chaque position
combine les valeurs disponibles. Ils ne doivent pas être interprétés
automatiquement comme une explication complète du raisonnement du
modèle.

### 5.3 Pourquoi l'attention doit-elle être causale ?

Dans un modèle qui prédit le prochain token, la position 3 ne doit pas
voir le token 4 pendant l'entraînement. Sinon, le modèle pourrait
tricher en regardant la réponse.

On applique donc un masque triangulaire : les positions futures sont
interdites.

``` python
T = 5
masque = torch.tril(torch.ones(T, T, dtype=torch.bool))
print(masque)
```

Une matrice de scores masquée peut être calculée ainsi :

``` python
scores = torch.randn(T, T)
scores = scores.masked_fill(~masque, float("-inf"))
poids = torch.softmax(scores, dim=-1)
print(poids)
```

Les positions futures ont un score de moins l'infini ; après softmax,
leur poids vaut zéro.

### 5.4 Une couche d'attention causale complète

``` python
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, block_size: int):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        )

    def forward(self, x):
        batch, time, channels = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        scores = q @ k.transpose(-2, -1) / (channels ** 0.5)
        scores = scores.masked_fill(
            ~self.mask[:time, :time],
            float("-inf")
        )
        weights = torch.softmax(scores, dim=-1)
        out = weights @ v
        return self.proj(out)
```

Cette version utilise une seule tête d'attention. Elle permet de
comprendre le calcul central avant de passer au multi-head attention.

**À retenir**

-   L'attention construit des mélanges pondérés d'informations.
-   Query et Key déterminent les poids ; Value transporte le contenu.
-   Le masque causal interdit de regarder les positions futures.
-   Les poids d'attention ne constituent pas à eux seuls une explication
    exhaustive du modèle.

------------------------------------------------------------------------
