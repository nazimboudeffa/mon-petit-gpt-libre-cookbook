# Troisième partie --- Construire et entraîner notre mini-GPT

## Chapitre 7 --- Le modèle complet

Nous assemblons maintenant les briques :

1.  embedding des tokens ;
2.  embedding des positions ;
3.  blocs Transformer ;
4.  normalisation finale ;
5.  projection vers le vocabulaire.

La dernière couche produit un score (*logit*) pour chaque token
possible.

### 7.1 Attention multi-têtes

``` python
class MultiHeadCausalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)
```

### 7.2 Bloc multi-têtes

``` python
class Block(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalAttention(
            d_model, n_heads, block_size, dropout
        )
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

### 7.3 Définir le GPT

``` python
class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[
            Block(d_model, n_heads, block_size, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError("La séquence dépasse block_size")

        positions = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(B * T, -1),
                targets.reshape(B * T)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0):
        self.eval()
        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
```

### 7.4 Compter les paramètres

``` python
model = MiniGPT(
    vocab_size=len(vocabulaire),
    block_size=64,
    d_model=128,
    n_heads=4,
    n_layers=2,
)

nombre_parametres = sum(p.numel() for p in model.parameters())
print(f"{nombre_parametres:,} paramètres")
```

Le nombre exact dépend de la taille du vocabulaire et de la
configuration. Il ne faut pas annoncer un chiffre sans préciser ces
paramètres.

------------------------------------------------------------------------
