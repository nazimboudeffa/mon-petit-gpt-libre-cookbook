# Annexe A --- Un script minimal pour vérifier les tenseurs

Avant de lancer un entraînement, il est utile de vérifier les
dimensions.

``` python
import torch

batch_size = 4
time_steps = 12
vocab_size = 20
d_model = 16

tokens = torch.randint(0, vocab_size, (batch_size, time_steps))
embedding = torch.nn.Embedding(vocab_size, d_model)
x = embedding(tokens)

print("tokens :", tokens.shape)  # (4, 12)
print("embeddings :", x.shape)   # (4, 12, 16)

projection = torch.nn.Linear(d_model, vocab_size)
logits = projection(x)

print("logits :", logits.shape)  # (4, 12, 20)
```

Les dimensions sont souvent la première source d'erreurs lors de
l'implémentation d'un réseau neuronal. Prenez l'habitude de les afficher
aux frontières des composants.
