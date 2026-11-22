# Première partie --- Du texte aux nombres

## Chapitre 1 --- Que fait réellement un modèle de langage ?

Un modèle de langage apprend à estimer quel token peut venir après une
séquence de tokens.

Prenons la phrase :

> Le chat dort sur le canapé.

Un modèle autorégressif apprend à prédire la suite à partir du préfixe.
Après « Le chat dort », il peut attribuer une probabilité à « sur », «
paisiblement », « dehors », ou à d'autres tokens de son vocabulaire.

Il ne reçoit pas directement des mots comme des objets compréhensibles.
Il reçoit des identifiants numériques. La première étape consiste donc à
convertir le texte en une suite d'entiers.

### 1.1 Prédire le prochain token

Soit une séquence de tokens :

$$
x_1, x_2, \ldots, x_t
$$

Le modèle cherche à estimer :

$$
P(x_{t+1} \mid x_1, x_2, \ldots, x_t)
$$

Il répète ensuite ce processus : le token généré est ajouté au contexte
et sert à prédire le suivant. C'est la génération autorégressive.

Cette formulation ne signifie pas que le modèle possède une
compréhension humaine. Elle décrit l'objectif mathématique
d'apprentissage : attribuer des probabilités aux tokens suivants.

### 1.2 Un premier exemple sans réseau neuronal

Avant d'utiliser PyTorch, observons une prédiction très simple fondée
sur les caractères.

``` python
from collections import defaultdict, Counter

texte = "bonjour bonjour bonsoir"
suivants = defaultdict(Counter)

for gauche, droite in zip(texte, texte[1:]):
    suivants[gauche][droite] += 1

caractere = "b"
print(suivants[caractere])
```

Ce programme compte les caractères qui suivent chaque caractère. Il ne
comprend pas la langue et ne prend en compte qu'un caractère de
contexte. Mais il illustre déjà une idée fondamentale : apprendre des
régularités à partir d'exemples.

### 1.3 Entraînement et génération sont deux opérations distinctes

Pendant l'entraînement, nous montrons au modèle des séquences dont nous
connaissons la suite attendue. Nous calculons une erreur entre les
prédictions et les bonnes réponses, puis nous ajustons les paramètres.

Pendant la génération, nous ne connaissons pas la bonne réponse. Le
modèle produit une distribution de probabilités, puis nous choisissons
un token selon une stratégie de sélection.

**À retenir**

-   Le modèle prédit une suite de tokens, pas directement des idées.
-   L'entraînement ajuste des paramètres à partir d'exemples.
-   La génération réutilise les prédictions précédentes comme nouveau
    contexte.

**Expérience**

Écrivez trois phrases très courtes et demandez-vous : après chaque
préfixe, quelles suites semblent plausibles ? Cette intuition de
prédiction du prochain élément est le point de départ du modèle.

------------------------------------------------------------------------
