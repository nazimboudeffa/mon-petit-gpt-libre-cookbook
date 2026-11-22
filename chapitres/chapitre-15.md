## Chapitre 15 --- Les prochaines améliorations techniques

Une fois les principes compris, plusieurs pistes deviennent accessibles
:

1.  remplacer le tokenizer caractère par un tokenizer BPE ;
2.  ajouter un token de début et un token de fin ;
3.  utiliser un jeu de validation et un jeu de test distincts ;
4.  enregistrer les courbes de loss ;
5.  ajouter un scheduler de taux d'apprentissage ;
6.  comparer plusieurs tailles de contexte ;
7.  implémenter le weight tying entre embeddings et tête de sortie ;
8.  utiliser l'attention optimisée de PyTorch lorsque la version le
    permet ;
9.  mesurer le temps et la mémoire consommés ;
10. documenter précisément les expériences.

Chaque amélioration doit répondre à une question. Ajouter une technique
parce qu'elle est à la mode ne remplace pas la compréhension de son
rôle.
