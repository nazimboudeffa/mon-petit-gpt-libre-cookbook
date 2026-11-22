## Chapitre 3 --- Construire les exemples d'entraînement

Un modèle autorégressif apprend à prédire le token suivant. À partir
d'une séquence, nous fabriquons donc deux tableaux décalés d'une
position.

Pour la séquence :

``` text
b o n j o u r
```

l'entrée est :

``` text
b o n j o u
```

et la cible est :

``` text
o n j o u r
```

À chaque position, le modèle apprend à prédire le caractère suivant.

### 3.1 Préparer un corpus

Créez un fichier `corpus.txt` contenant quelques paragraphes originaux.
Pour le premier essai, le corpus peut être très petit :

``` text
Le petit robot observe le jardin.
Le jardin est calme au lever du soleil.
Le robot apprend à nommer les fleurs.
Une abeille passe au-dessus des marguerites.
Le robot continue d'observer le jardin.
```

Un corpus minuscule est suffisant pour vérifier le pipeline, mais pas
pour apprendre à écrire correctement en français.

### 3.2 Charger et encoder le texte

``` python
from pathlib import Path

texte = Path("corpus.txt").read_text(encoding="utf-8")
vocabulaire = sorted(set(texte))

token_vers_id = {c: i for i, c in enumerate(vocabulaire)}
id_vers_token = {i: c for c, i in token_vers_id.items()}

donnees = [token_vers_id[c] for c in texte]
print("Caractères :", len(texte))
print("Vocabulaire :", len(vocabulaire))
```

### 3.3 Créer les fenêtres de contexte

Nous fixons une longueur maximale de contexte `block_size`. Chaque
exemple contient une fenêtre d'entrée et sa cible décalée.

``` python
import torch

block_size = 32
x_exemples = []
y_exemples = []

for debut in range(len(donnees) - block_size):
    x_exemples.append(donnees[debut:debut + block_size])
    y_exemples.append(donnees[debut + 1:debut + block_size + 1])

x = torch.tensor(x_exemples, dtype=torch.long)
y = torch.tensor(y_exemples, dtype=torch.long)

print(x.shape, y.shape)
```

Pour chaque ligne, `x` et `y` ont la même longueur. La cible de chaque
position correspond au token suivant.

### 3.4 Séparer entraînement et validation

Il est important de réserver une partie des données pour la validation.
Le modèle ne doit pas apprendre sur ces exemples pendant la mise à jour
de ses paramètres.

``` python
n = int(0.9 * len(donnees))
train_ids = torch.tensor(donnees[:n], dtype=torch.long)
val_ids = torch.tensor(donnees[n:], dtype=torch.long)
```

Cette séparation est simple et respecte l'ordre du texte. Pour un corpus
très petit, les mesures resteront instables. Dans un projet sérieux, la
séparation doit aussi éviter les fuites entre documents quasi
identiques.

**À retenir**

-   L'entrée et la cible sont la même séquence décalée d'un token.
-   Le modèle apprend simultanément à plusieurs positions.
-   La validation aide à détecter la mémorisation, mais un petit jeu de
    validation ne suffit pas à prouver la généralisation.

------------------------------------------------------------------------
