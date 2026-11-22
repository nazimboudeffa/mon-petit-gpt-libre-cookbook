## Chapitre 6 --- Multi-head attention et bloc Transformer

Une seule tête d'attention apprend un seul ensemble de relations dans un
espace de représentation. Le multi-head attention exécute plusieurs
mécanismes d'attention en parallèle, puis rassemble leurs résultats.

Dans une implémentation courante, la dimension totale est répartie entre
plusieurs têtes. Chaque tête possède ses propres projections.

### 6.1 Le réseau feed-forward

Après l'attention, chaque position passe par un petit réseau neuronal
appliqué indépendamment aux positions :

$$
\operatorname{FFN}(x) = W_2 \sigma(W_1 x + b_1) + b_2
$$

Il transforme les caractéristiques calculées par l'attention.

### 6.2 Connexions résiduelles et normalisation

Un bloc Transformer utilise des connexions résiduelles. Au lieu de
remplacer directement la représentation, le sous-réseau apprend une
modification à lui ajouter.

La normalisation aide à stabiliser les activations. Dans ce livre, nous
utilisons une architecture Pre-LayerNorm, où la normalisation intervient
avant les sous-couches.

### 6.3 Implémenter un bloc

``` python
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.ffn(self.ln2(x))
        return x
```

Pour garder ce bloc facile à lire, l'attention est encore à une tête. Le
chapitre suivant proposera une version multi-têtes intégrée au modèle
complet.

------------------------------------------------------------------------
