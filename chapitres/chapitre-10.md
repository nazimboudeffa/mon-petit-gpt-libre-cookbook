## Chapitre 10 --- Générer du texte

Une fois entraîné, le modèle peut produire de nouveaux tokens à partir
d'un préfixe.

``` python
depart = torch.zeros((1, 1), dtype=torch.long, device=device)
sortie = model.generate(depart, max_new_tokens=300, temperature=0.8)

texte_genere = "".join(
    id_vers_token[i] for i in sortie[0].tolist()
)
print(texte_genere)
```

Ici, l'identifiant zéro est utilisé comme point de départ par commodité.
Dans un système plus complet, on définirait un token de début dédié.

### 10.1 Température

La température modifie la distribution avant l'échantillonnage :

-   température basse : les choix les plus probables sont favorisés ;
-   température élevée : la distribution est plus étalée ;
-   température proche de zéro : le comportement devient très
    déterministe, sous réserve de la stratégie de sélection.

La température ne rend pas le modèle plus intelligent. Elle modifie la
diversité des sorties.

### 10.2 Pourquoi les résultats peuvent sembler absurdes

Le modèle apprend les régularités de son corpus. Si le corpus ne
contient que cinq phrases, il dispose de très peu d'informations. Il
peut reproduire des fragments, mélanger des motifs ou générer des
séquences incohérentes.

C'est un résultat pédagogique intéressant : il montre qu'une
architecture GPT ne suffit pas. Les données, la taille du modèle,
l'entraînement et l'évaluation comptent énormément.

------------------------------------------------------------------------
