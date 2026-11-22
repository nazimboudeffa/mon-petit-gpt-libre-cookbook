## À propos de ce livre

Ce livre propose de comprendre les grands modèles de langage en
construisant un petit GPT avec Python et PyTorch. Le but n'est pas
d'obtenir un assistant aussi performant qu'un service commercial, mais
de rendre visibles les mécanismes qui se cachent derrière une génération
de texte.

Nous allons partir de rien : un petit corpus, un tokenizer caractère,
des embeddings, un mécanisme d'attention causale, des blocs Transformer,
une fonction de perte et une boucle d'entraînement. À la fin, nous
disposerons d'un modèle minuscule capable de produire du texte
ressemblant à ses exemples d'entraînement.

Le fil rouge est volontairement simple. Un modèle de quelques milliers
ou centaines de milliers de paramètres permet d'inspecter les tenseurs,
de modifier le code et de comprendre les conséquences de chaque choix.
Il ne faut pas confondre cette simplicité avec les systèmes industriels
: ceux-ci reposent sur des corpus immenses, des infrastructures
distribuées, des tokenizers plus élaborés, des évaluations rigoureuses
et de nombreuses étapes supplémentaires.

### Ce que vous saurez faire

À la fin de la lecture, vous devriez pouvoir :

-   expliquer la différence entre tokenisation, embedding et génération
    ;
-   décrire le rôle des Query, Key et Value ;
-   expliquer pourquoi l'attention est causale dans un modèle
    autorégressif ;
-   identifier les composants d'un bloc Transformer ;
-   entraîner un petit modèle avec une boucle PyTorch ;
-   lire une courbe de loss sans lui faire dire plus qu'elle ne dit ;
-   reconnaître les limites d'un modèle entraîné sur un minuscule
    corpus.

### Prérequis

Une connaissance élémentaire de Python est utile : variables, fonctions,
listes, boucles et imports. Il n'est pas nécessaire de connaître déjà
les Transformers. Une familiarité avec les tenseurs et les
multiplications de matrices facilitera certains passages, mais les
notions seront introduites progressivement.

### Environnement

Les exemples ciblent Python 3.10 ou plus récent et PyTorch. Installez
PyTorch depuis la page officielle correspondant à votre système et à
votre matériel : <https://pytorch.org/get-started/locally/>.

Pour vérifier l'installation :

``` bash
python -c "import torch; print(torch.__version__)"
```

Les exemples sont conçus pour fonctionner sur CPU. Un GPU peut accélérer
l'entraînement, mais n'est pas indispensable pour comprendre le modèle.

------------------------------------------------------------------------
