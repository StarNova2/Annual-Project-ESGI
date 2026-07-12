## Exp 1
- utilisation de valuer pris au hasard 
  - parametre
    - mouvement max : 0.0001
    - nb cluster : 200
    - max loop : 1000
    - gamma : 0.01
    - liste seed :[12, 24, 42, 5, 59]
  - résultat
    - loss quasi équivalent (train test) -> moyenne de 0,2
    - accuracy : 0,9 moy
  - obsetvation
    - bonne accuracy, même avec un gamma petitles cluster semble bien placé
    - Je garde cette version pour avoir une bonne base, et je part au début pour comprendre les influence de chaque paramètre
    
## Exp 2
- partons pour "essayer", sur 1, 2 et 3 cluster par modèle avec le même gamma qu'avant pour voir quelle résultats ça donne  
  - parametre
    - mouvement max : 0.0001
    - nb cluster : 1, 2, 3
    - max loop : 1000
    - gamma : 0.01
    - liste seed :[12, 24, 42, 5, 59]
  - résultat
    - loss  -> meilleur loss  de 0,9 3 cluster
    - accuracy : 0.63 max
  - obsetvation
    - Le loss est toujours désastreux pour 1 ou 2 cluster (aléatoire mieux)
    - comme le dataset contient plusieurs jeu, on pourrais se dire que 3 cluster représnete 3 jeux mageur du dataset
    - prochain test avec 4 pourvoir si une amélioration est remarqué, puis commencer à jouer avec le gamma


## Exp 3   
- regardons si 4 cluster augmente l'accuracy
  - parametre
    - mouvement max : 0.0001
    - nb cluster : 4
    - max loop : 1000
    - gamma : 0.01
    - liste seed :[12, 24, 42, 5, 59]
  - résultat
    - loss  -> meilleur loss  de 0,9 
    - accuracy : 0.63 moy
  - obsetvation
    - les stats ne semble pas trop changer, même accuracy, même loss, même matrice
    - faisons


## Corps 
- utilisation  
  - parametre
    - mouvement max : 
    - nb cluster : 
    - max loop : 
    - gamma : 
    - liste seed :[12, 24, 42, 5, 59]
  - résultat
    - loss  -> moyenne de 0,2
    - accuracy :  moy
  - obsetvation