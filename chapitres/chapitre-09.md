## Chapitre 9 --- Entraîner le modèle

L'entraînement suit une boucle répétée :

1.  prendre un lot d'exemples ;
2.  calculer les prédictions ;
3.  calculer la perte ;
4.  remettre les gradients à zéro ;
5.  calculer les gradients par rétropropagation ;
6.  mettre à jour les paramètres avec l'optimiseur.

### 9.1 Échantillonner des lots

``` python
def get_batch(data, batch_size, block_size, device):
    starts = torch.randint(
        0, len(data) - block_size - 1, (batch_size,)
    )
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
```

### 9.2 Boucle d'entraînement

``` python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

batch_size = 32
max_steps = 1000

for step in range(max_steps):
    model.train()
    xb, yb = get_batch(train_ids, batch_size, model.block_size, device)

    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 100 == 0:
        print(f"étape {step:4d} | loss {loss.item():.4f}")
```

Cette boucle est volontairement concise. Pour une expérience
reproductible, il faudra fixer les graines aléatoires, enregistrer la
configuration, mesurer la validation et sauvegarder les poids.

### 9.3 Évaluer sur la validation

``` python
@torch.no_grad()
def evaluate(model, data, batch_size=32, batches=20):
    model.eval()
    losses = []

    for _ in range(batches):
        xb, yb = get_batch(data, batch_size, model.block_size, device)
        _, loss = model(xb, yb)
        losses.append(loss.item())

    return sum(losses) / len(losses)

print("Validation :", evaluate(model, val_ids))
```

Si la perte d'entraînement diminue alors que celle de validation
augmente, le modèle peut être en train de mémoriser les exemples
d'entraînement. Sur un corpus minuscule, ce phénomène est fréquent.

**Attention :** dans cet exemple minimal, `train_ids` et `val_ids`
doivent chacun contenir plus de `block_size + 1` tokens. Vérifiez cette
condition avant de lancer l'entraînement.

------------------------------------------------------------------------
