## Chapitre 13 --- Sauvegarder et recharger le modèle

Pour conserver les paramètres appris, on sauvegarde le `state_dict`.

``` python
torch.save(
    {
        "model_state": model.state_dict(),
        "vocabulaire": vocabulaire,
        "config": {
            "block_size": model.block_size,
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 2,
        },
    },
    "mini_gpt.pt",
)
```

Pour recharger, il faut reconstruire le modèle avec exactement la même
architecture et le même vocabulaire :

``` python
checkpoint = torch.load("mini_gpt.pt", map_location=device, weights_only=False)

config = checkpoint["config"]
vocabulaire = checkpoint["vocabulaire"]

model = MiniGPT(
    vocab_size=len(vocabulaire),
    block_size=config["block_size"],
    d_model=config["d_model"],
    n_heads=config["n_heads"],
    n_layers=config["n_layers"],
).to(device)

model.load_state_dict(checkpoint["model_state"])
model.eval()
```

Le paramètre `weights_only` dépend de la version de PyTorch. Ne chargez
jamais un fichier de checkpoint provenant d'une source non fiable : un
fichier de sérialisation peut présenter des risques selon son format et
la méthode de chargement.

Pour un projet propre, enregistrez également la version de Python, celle
de PyTorch, les graines, les hyperparamètres, la provenance des données
et le commit du code.

------------------------------------------------------------------------
