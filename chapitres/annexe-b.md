# Annexe B --- Dépannage

### `ModuleNotFoundError: No module named 'torch'`

PyTorch n'est pas installé dans l'environnement Python actif. Vérifiez
l'interpréteur utilisé et installez PyTorch dans cet environnement.

### `IndexError: index out of range in self`

Un identifiant dépasse la taille du vocabulaire ou une position dépasse
`block_size`. Vérifiez les identifiants et les dimensions.

### `Expected input batch_size ...`

La forme des logits et celle des cibles ne correspondent pas. Pour
`cross_entropy`, aplatissez les dimensions batch et temps comme dans le
code du modèle.

### La loss devient `nan`

Vérifiez le taux d'apprentissage, les données, les logits et les
gradients. Réduisez le taux d'apprentissage et inspectez les tenseurs
avec `torch.isfinite`.

### Le modèle ne produit que des caractères répétitifs

Le corpus peut être trop petit, l'entraînement trop long, le modèle mal
réglé ou la génération trop déterministe. Vérifiez d'abord que le
pipeline apprend sur un corpus de test simple.
