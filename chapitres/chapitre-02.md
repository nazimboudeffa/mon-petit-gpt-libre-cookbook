## Chapitre 2 --- Tokeniser : transformer du texte en identifiants

Un réseau neuronal manipule des nombres. Il faut donc définir un
vocabulaire et une conversion réversible entre tokens et identifiants.

Pour commencer, nous utiliserons un tokenizer caractère. Chaque
caractère distinct devient un token. C'est simple, mais peu efficace
pour des textes réels : un mot peut occuper plusieurs tokens et les
séquences deviennent longues.

### 2.1 Construire un vocabulaire

``` python
texte = "bonjour le monde"

vocabulaire = sorted(set(texte))
token_vers_id = {token: i for i, token in enumerate(vocabulaire)}
id_vers_token = {i: token for token, i in token_vers_id.items()}

print(vocabulaire)
print(token_vers_id)
```

L'encodage transforme le texte en entiers :

``` python
def encoder(texte: str) -> list[int]:
    return [token_vers_id[c] for c in texte]

def decoder(ids: list[int]) -> str:
    return "".join(id_vers_token[i] for i in ids)

ids = encoder("bonjour")
print(ids)
print(decoder(ids))
```

Le vocabulaire est construit à partir du corpus. Dans un projet réel, il
faut aussi prévoir les caractères inconnus, les espaces, les caractères
Unicode et une politique de normalisation.

### 2.2 Pourquoi utiliser BPE dans les grands modèles ?

Les tokenizers modernes utilisent souvent des sous-mots. Une méthode
connue est Byte Pair Encoding (BPE). L'idée générale est de fusionner
progressivement les paires d'unités fréquentes afin de construire un
vocabulaire de morceaux de texte.

Un tokenizer sous-mots peut représenter un mot courant en un seul token,
tout en décomposant un mot rare en plusieurs morceaux. Cela réduit
souvent la longueur des séquences par rapport à un tokenizer caractère.

Nous gardons le tokenizer caractère pour que le code reste lisible. Le
but du livre est d'abord de comprendre le Transformer, pas de
réimplémenter intégralement un tokenizer industriel.

### 2.3 Attention aux identifiants

Les identifiants n'ont pas de signification numérique intrinsèque. Le
token 8 n'est pas « plus grand » ou « plus important » que le token 2.
Ce sont des étiquettes permettant d'indexer une table d'embeddings.

**À retenir**

-   Le tokenizer définit l'unité que le modèle manipule.
-   Le vocabulaire associe chaque token à un entier.
-   Le tokenizer caractère est pédagogique, mais rarement le meilleur
    choix pour un grand corpus.

**Expérience**

Affichez la longueur du texte et le nombre de tokens. Essayez avec un
texte français contenant des accents, des apostrophes et des emojis.
Vérifiez que votre encodeur et votre décodeur préservent bien le texte.

------------------------------------------------------------------------
