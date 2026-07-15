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
    - les stats ne semble pas trop changer, même accuracy, même loss, quasi même matrice
    - essayons de baisser et augmenter le gamma 



## Exp 4   
- regardons si 4 cluster augmente l'accuracy
  - parametre
    - mouvement max : 0.0001
    - nb cluster : 4
    - max loop : 1000
    - gamma : 0.0001, 0.001, 0.1, 1
    - liste seed :[12, 24, 42, 5, 59]
  - résultat
    - loss : ereur, (train -> 0.8, test -> 0.75), ~1, 2.5
    - accuracy : erreur, 0.6, 0.6, 0.02
  - obsetvation
    - pour 4 cluster, on vois que plus le gamma augmente, moins il est précis
    - Plus le gamma est grand, plus la gaussienne sera "pointu", donc avec 1; le cluster ne peut pas différencié les données, alors qu'avec un loss petit, le cluster pourrais voir l'entierté de l'image
    - Donc peut être qu'avec peu de cluster, le gamma devra être petit, mais avec bcp de cluster, le gamma devra être grand
    - Je vais passer à 6, 8 et 10 cluster avec différent gamma pour voir s'il en avoir un petit est intéressant 


## Exp 5   
- nb cluster 6, 8, 10 avec diffrénet gamma pour validé une partie de ma théorie
  - parametre
    - mouvement max : 0.0001
    - nb cluster : 6, 8, 10
    - max loop : 1000
    - gamma : 0.0001
    - liste seed :[12, 24, 42, 5, 59]
  - résultat
    - loss : 6(train : 0.7, test : 0.65), 8(erreur(gamma trop petit)  ), 10(erreur(gamma trop petit) ou 0.65 / 0.9)
    - accuracy : 6(0.62), 8(erreur(gamma trop petit)), 10(erreur(gamma trop petit) ou 0.5 de moy)
  - obsetvation
    - à pemière vu, avec 8 et 10 cluster le gamma trop petit faisait crashé le modèle. Mais après avoir essayé avec d'autre seed, le cluster 10 à réussi à donner un beau résultats en loss. Mais sur 3 seed, il y avais 1 résultats assez mauvais (0.9 de loss, 0.4 accuracy)
    - les cluster étant plus nombreux, et ayant une grande guaussienne, les cluster se ressemblais ce qui à fait planté le modèle, donc ça sera compliqué de remettre 0.0001 en gamma plus tard 
    - continuons sur notre lancé 6, 8, 10 avec le gamma 0.001. Est ce que 6 cluster avec un plus grand gamma serait iontéressant ? normalement non 


## Exp 6
- nb cluster 6, 8, 10 avec diffrénet gamma pour validé une partie de ma théorie
  - parametre
    - mouvement max : 0.0001
    - nb cluster : 6, 8, 10
    - max loop : 1000
    - gamma : 0.001
    - liste seed :[12, 24, 42, 5, 59]
  - résultat
    - loss : 6(train : 0.65 , test : 0.6 ), 8(train : 0.65 , test : 0.55), 10(train : 0.6 , test : 0.5)
    - accuracy : 6(0.7), 8(0.68), 10(0.7)
  - obsetvation
    - un peu d'amélioration, continuons avec 0.01


## Exp 7
- nb cluster 6, 8, 10 avec diffrénet gamma pour validé une partie de ma théorie
  - parametre
    - mouvement max : 0.0001
    - nb cluster : 6, 8, 10
    - max loop : 1000
    - gamma : 0.01
    - liste seed :[12, 24, 42, 5, 59]
  - résultat
    - loss : 6(train : 0.75 , test : 0.65 ), 8(train : 0.65 , test : 0.55), 10(train : 0.6 , test : 0.55)
    - accuracy : 6(0.67), 8(0.68), 10(0.74)
  - obsetvation
    - un peu d'amélioration
    - matrice : on vois bien que la plus part des jeu sont trié en majorité. Je pense qu'un type de scren ressort dans les FPS alors que se sont des moba, mais moba sur téléphone
         FPS  METRO   MOBA
  FPS    105      1     14
METRO     11    132      6
 MOBA     34      0     88
    - on commence à stagner au niveau résultats, passons à la vitesse supérieur avec un +10 en nb cluster avec plusieurs gamma qui tend vers le plus pour voir si les cluster arrive à voir certaine zone


## Exp 8
- nb cluster 20, 30, 40 -> essayon de voir si avec plus de cluster il y a plus de résultats de plus en faisant varié le gamma vers le haut
  - parametre
    - mouvement max : 0.0001
    - nb cluster : 20, 30, 40
    - max loop : 1000
    - gamma : 0.001
    - liste seed :[12, 24, 42, 5, 59]
  - résultat
    - loss : 20(train : 0.4, test : 0.38), 30(train : 0.35, test :0.3)m 40(erreur + train: 0.25, test : 0.25 )
    - accuracy : 20(0.79), 30(0.83), 40(erreur + 0.86)
    - matrice 20:             
         FPS  METRO   MOBA
  FPS    105      2     13
METRO      7    139      3
 MOBA     25      0     97
    - 30 
         FPS  METRO   MOBA
  FPS    111      4      5
METRO      7    139      3
 MOBA     16      0    106
    - 40
         FPS  METRO   MOBA
  FPS    109      4      7
METRO      7    141      1
 MOBA      9      0    113

  - obsetvation
    - les test sont bien plus concluent 
    - pour 40 cluster, 0.001 de gamma ne suffisent presque plus
    - l'accuracy reste mieux pour 40 cluster avec 0.001 de gamma, c'est la limite
    - au nivea des matrice, la plus part des moba sont trié mais bloqué en fps, je viens de voir que certaine images sont mal prises dans le dataset
    - Je vais les retirer pour voir si ça à un impact
  


## Exp 9
- nb cluster 20, 30, 40 -> avec dataset plus clean
  - parametre
    - mouvement max : 0.0001
    - nb cluster : 20, 30, 40
    - max loop : 1000
    - gamma : 0.001
    - liste seed :[12, 24, 42, 5, 59]
  - résultat
    - loss : 20(erreur + train : 0.42, test : 0.38), 30(train : 0.33, test :0.3)m 40(erreur + train: 0.25, test : 0.25 )
    - accuracy : 20(erreur + 0.78), 30(erreur + 0.83), 40(erreur + 0.86)
    - matrice 20:             
         FPS  METRO   MOBA
  FPS     97      4      6
METRO      7    139      2
 MOBA     30      0     88
    - 30 
         FPS  METRO   MOBA
  FPS    111      4      5
METRO      7    139      3
 MOBA     16      0    106
    - 40
    FPS  METRO   MOBA
  FPS    100      5      2
METRO      6    141      1
 MOBA     17      0    101

  - obsetvation
    - même avec des données plus clean, l'accuracy et loss sont quasiment identique
    - par contre, on rencontre des erreurs comparer à avant pour 20 et 30 cluster, donc les données retiré avait bien un apport d'information qui faisait varier les clusters. Donc maintenant, les cluster se ressemblait plus avec ce gamma
    - au niveau des matrices, il n'y a pas de grand changement, donc à première vu, se ne sont pas les données retiré qui impact la matrice
    - Après discussion avec mon équipe, nous resterons sur le dataset nettoyé

  

## Exp 10
- nb cluster 20, 30, 40 -> essayon de voir si avec plus de cluster il y a plus de résultats de plus en faisant varié le gamma vers le haut
  - parametre
    - mouvement max : 0.0001
    - nb cluster : 20, 30, 40
    - max loop : 1000
    - gamma : 0.01
    - liste seed :[12, 24, 42, 5, 59]
  - résultat
    - loss : 20(train : 0.45, test : 0.40), 30(train : 0.38, test :0.35)m 40(train: 0.35, test : 0.0.33 )
    - accuracy : 20(0.79), 30(0.81), 40(0.83)
    - matrice 20:             
          FPS  METRO   MOBA
  FPS     90      4     14
METRO      4    140      4
 MOBA     19      0     99
    - 30 
         FPS  METRO   MOBA
  FPS     94      5      9
METRO      9    137      2
 MOBA     23      1     94
    - 40
         FPS  METRO   MOBA
  FPS     90      7     11
METRO      8    140      0
 MOBA     23      0     95

  - obsetvation
    - au niveau du loss et de l'accuracy, on observe bien que plusle gamma est petit, plus les résultats sont intéressant 
    - dernier test pour confirmer la théorie, néanmoins, on avais vu sur les cas de test qu'un petit gamma pouvais ne pas fonctionner, il en fallais un plus gros. Donc i lne faut pas écarter la possibilité d'augmenter le gamma


## Exp 11
- nb cluster 20, 30, 40 -> essayon de voir si avec plus de cluster il y a plus de résultats de plus en faisant varié le gamma vers le haut
  - parametre
    - mouvement max : 0.0001
    - nb cluster : 20, 30, 40
    - max loop : 1000
    - gamma : 0.1
    - liste seed :[12, 24, 42, 5, 59]
  - résultat
    - loss : 20(train : 0.78, test : 0.7), 30(train : 0.63, test :0.6)m 40(train: 0.65, test : 0.63 )
    - accuracy : 20(0.6), 30(0.68), 40(0.72)
    - matrice 20:             
          FPS  METRO   MOBA
  FPS     62      3     43
METRO     26    117      5
 MOBA      8      1    109
    - 30 
         FPS  METRO   MOBA
  FPS     66      3     39
METRO     11    123     14
 MOBA     11      0    107
    - 40
         FPS  METRO   MOBA
  FPS     74      9     25
METRO      6    127     15
 MOBA     11      2    105

  - obsetvation
    - confirmation de la première théory -> plus gamma est petit, mieux c'est 



### Question :
- de combien de cluster on devrais avoir besoin en théory ? 
  - plus il y a de cluster mieux c'est, mais trop de cluster ne servirais plus à un moment si on reprend la logique
  - essayons avec beaucoup de cluster en réduisnat le gamma petit à petit, puis on feras un -100 cluster pour voir si les résultats ont une grosse diffrénces


## Exp 12
- nb cluster 1000, gamma 1 vers -
  - parametre
    - mouvement max : 0.0001
    - nb cluster : 1000
    - max loop : 1000
    - gamma : 1, 0.1, 0.01
    - liste seed :[12, 24, 42, 5, 59]
  - résultat
    - loss : 1(train: 0.75, test : 2), 0.1(train: 0.2, test: 0.35), 0.01(erreur)
    - accuracy : 1(0.2), 0.1(0.8), 0.01(erreur)
    - matrice 1:             
          FPS  METRO   MOBA
  FPS    107      0      1
METRO    106     42      0
 MOBA     59      4     55
    - 0.1 
         FPS  METRO   MOBA
  FPS     88      2     18
METRO      6    137      5
 MOBA      7      0    111
  - obsetvation
    - une erreur est survenu pour des gamma plus petit, mais nous avons un résultat intéressant avec 0.1 de gamma, mais passer de 40 cluster à 1000 pour des résultats équivalent montre qu'on en à pas besoin d'énormément 
    - testons avec 900 cluster pour voir les différence



## Exp 13
- nb cluster 900, gamma 1 vers -. Obj -> regarder la différence entre 1000 et 900 cluster pour voir si ça sert d'avoir bcp de cluster
  - parametre
    - mouvement max : 0.0001
    - nb cluster : 900
    - max loop : 1000
    - gamma : 1, 0.1, 0.05
    - liste seed :[12, 24, 42, 5, 59]
  - résultat
    - loss : 1(train: 0.75, test : 2), 0.1(train: 0.2, test: 0.35), 0.05(trzin : 0.2, test : 0.33 )
    - accuracy : 1(0.2), 0.1(0.8), 0.05(0.83)
    - matrice 1:             
          FPS  METRO   MOBA
  FPS    107      0      1
METRO    106     42      0
 MOBA     59      4     55
    - 0.1 
         FPS  METRO   MOBA
  FPS     88      2     18
METRO      6    137      5
 MOBA      7      0    111
  - 0.05
         FPS  METRO   MOBA
  FPS     96      4      8
METRO      8    137      3
 MOBA     10      0    108
  - obsetvation
    - on peut voir qu'il y a une très légère différence entre les résultats avec 900 et 1000 cluster, donc l'utilisation de trop de cluster n'ai pas une bonne approche


### Réflexion
- Maintenant, il serais intéressant de voir combien de cluster il faudrais en théorie
- les cluster vons prendre aléatoirement des regroupenment de données, donc sur une image comme overwatch, il y a :
  - profil joueur 
  - barre de point de vie
  - logos des compétences 
  - visuel de l'arme dans barre de compétence
  - logo de la compétence spécial 
  - timer
  - information sur le mod de jeu (logo chargement, barre de progression, score, ...)
  - arme du personnage
- Cependant, ces informations peuvent prendre beaucoup d'espace, et peuvent changer de couleurs, donc pour Overwatch, je pourrais mettre théoriquement :
  - profil joueur (~36 avec skin -> non quantifiable) --> 1 cluster
  - barre de point de vie (3~4 colori de point de vie avec une grande taille) --> 6 cluster max ?
  - logos des compétences (allant de 2 à 5 avec 3 coloris différénet) --> 15 cluster max ?  
  - visuel de l'arme dans barre de compétence (une arme par perso (~36)) --> 3 cluster ? 
  - logo de la compétence spécial  (chargé ou non chargé, donc 2 colori) --> 2 cluster
  - timer + score (longue barre) -->2 cluster
  - information sur le mod de jeu (logo chargement, barre de progression, score, ...) (change beaucoup) -->20 cluster
  - arme du personnage (une arme par perso (~36 sans parler des skin), prend de la place) --> 50 cluster ? 

- pour overwatch, nous avons potentielement 99 cluster, mais overwatch n'est pas le seul jeu 
- on pourrais partir sur une base d'environ 100 cluster par jeu vraiment différent (moba, fps, metroidvania) --> 300
- on iras plus loin car même dans des jeu de la même catégorie, il peuvent ne pas se ressembler, ou avoir les logo autre part 
- peut être ajouté après 50 cluster 
- essayons 100, 200, 300 


## Exp 14
- utilisation  100, 200, 300
  - parametre
    - mouvement max : 0.0001
    - nb cluster : 100, 200, 300
    - max loop : 1000 
    - gamma : 0.01
    - liste seed :[12, 24, 42, 5, 59]
  - résultat
    - loss  : 100(train: 0.25 , test: 0.2), 200(train: 0.2 , test: 0.19), 300(erreur + test : 0.14, train : 0.17)
    - accuracy :  100(0.89), 200(0.9), 300(erreur + 0.9)
    - matrice 100:             
          FPS  METRO   MOBA
  FPS    100      4      4
METRO      5    142      1
 MOBA     14      0    104
    - 200:             
         FPS  METRO   MOBA
  FPS    101      4      3
METRO      3    144      1
 MOBA     11      0    107
    - 300
         FPS  METRO   MOBA
  FPS    103      3      2
METRO      5    143      0
 MOBA     13      0    105
  - obsetvation
    - on vois que les perf sont très correcte et même plus intéressant que 900 cluster
    - on vois qu'il y a toujours un problème avec les moba qui sont (pour une partie) vue comme des moba
    - continuons avec un gamma plus grand pour avoir 300 cluster 
    - je ne testerais pas 100 cluster car nous avons vue qu les résultats sont rarement au rendez-vous





## Exp 15
- utilisation 200, 300
  - parametre
    - mouvement max : 0.0001
    - nb cluster : 100, 200, 300
    - max loop : 1000 
    - gamma : 0.1
    - liste seed :[12, 24, 42, 5, 59]
  - résultat
    - loss  :  200(train: 0.2 , test: 0.19), 300(test : 0.4, train : 0.42)
    - accuracy :  200(0.9), 300(0.7)
    - matrice 200
         FPS  METRO   MOBA
  FPS    101      4      3
METRO      3    144      1
 MOBA     11      0    107
    - 300
          FPS  METRO   MOBA
  FPS     80      7     21
METRO      5    136      7
 MOBA      6      0    112
  - obsetvation
    - moins de résultats, je devrais garder 300 cluster avec ~0.01 de gamma
    - on vois qu'il y a toujours un problème avec les moba qui sont (pour une partie) vue comme des moba
    - je vais essayer d'augmenter le nombre de cluster si jamais on peut avoir de meilleur résulta


## Exp 16
- utilisation 350
  - parametre
    - mouvement max : 0.0001
    - nb cluster : 350
    - max loop : 1000 
    - gamma : 0.01
    - liste seed :[12, 24, 42, 5, 59]
  - résultat
    - loss  :  350(erreur + test : 0.13, train : 0.17)
    - accuracy :  350(erreur + 0.9)
    - matrice 350
          FPS  METRO   MOBA
  FPS    102      4      2
METRO      5    143      0
 MOBA     13      0    105

  - obsetvation
    - 350 cluster est intéressant
    - 90% d'accuracy, un loss de 0.13 et 0.17
    - avnt il y avait en bon cluster : 300 cluster, 90% d'accuracy et 0.14 et 0.17 de loss
    - les tests en modifiant le mouvement_max et max_loop ne donne pas de changement aux résultats


### Blocage
- J'ai essayé bien plus de cluster (4000), avec des gamma variant de 20 à 0.1, changeant el mouvemen_max ou max loop, 
- les résultats en loss reste supérieur à 0.2
- les résultats en accuracy était inférieur à 80%
- je réalise après test avec 400 cluster, je tombe toujours sur l'erreur des moba
- je décide de passer à 500 avec un petit gamma qui rend le modèle fragile, mais l'erreur est réduite 

## Exp 16
- utilisation 500
  - parametre
    - mouvement max : 0.0001
    - nb cluster : 500
    - max loop : 1000 
    - gamma : 0.01
    - liste seed :range(0, 100) (14)
  - résultat
    - loss  :  test : 0.11, train : 0.15
    - accuracy :  0.92
    - matrice 
          FPS  METRO   MOBA
  FPS    102      4      2
METRO      2    146      0
 MOBA      9      0    109

  - obsetvation
    - on est en train de gratter des résultats pour potentiellement faire du sur-apprentissage
    - après quelque test, on dirait que le modèle n'a pas de sur-apprentissage, je vais donc encore creuser pour avoir plus de cluster
    - je vais aussi partir sur le modèle à 200 ou 300 cluster si jamais celui à 500 sur-apprend


## Exp 17
- utilisation 500 cluster pour avoir le meilleur
  - parametre
    - mouvement max : 0.0001
    - nb cluster : 500
    - max loop : 1000 
    - gamma : 0.01
    - liste seed : [14, 1, 2, 12, 59, 96, 3, 11, 13, 18, 26, 30, 59] best -> 3
  - résultat
    - loss  :  test : 0.11, train : 0.13
    - accuracy :  0.938
    - matrice 3
          FPS  METRO   MOBA
  FPS    102      4      2
METRO      2    146      0
 MOBA      9      0    109

  - obsetvation
    - essayons plus petit gamma


## Exp 18
- utilisation 500 avec moins de gamma
  - parametre
    - mouvement max : 0.0001
    - nb cluster : 500
    - max loop : 1000 
    - gamma : 0.0095
    - liste seed : [14, 1, 2, 12, 59, 96, 3, 11, 13, 18, 26, 30, 59] best -> 59
  - résultat
    - loss  :  test : 0.09, train : 0.14
    - accuracy :  0.93
    - matrice 59
          FPS  METRO   MOBA
  FPS    102      5      1
METRO      6    142      0
 MOBA      9      0    109
  - obsetvation
    - le résultats ne change pas trop, essayons avec plus de cluster 


## Exp 19
- utilisation 510, 520, 550, 530
  - parametre
    - mouvement max : 0.0001
    - nb cluster : 510, 
    - max loop : 1000 
    - gamma : 0.01
    - liste seed : [14, 1, 2, 12, 59, 96, 3, 11, 13, 18, 26, 30, 59] best -> 2
  - résultat
    - loss  :  510(test : 0.09, train : 0.14), 550(pareil), 520(test : 0.11, test : 0.13), 530(0.1, 0.15)m 524(0.11, 0.13)
    - accuracy :  510(0.93), 550(0.92), 530(0.92), 520(0.939)
    - matrice 2 -> 510
         FPS  METRO   MOBA
  FPS    101      5      2
METRO      4    144      0
 MOBA      9      0    109
    - 550
         FPS  METRO   MOBA
  FPS    102      5      1
METRO      4    143      1
 MOBA     12      0    106
    - 520
         FPS  METRO   MOBA
  FPS    103      4      1
METRO      2    146      0
 MOBA      9      0    109
    - 530
         FPS  METRO   MOBA
  FPS    102      4      2
METRO      6    142      0
 MOBA     11      0    107
    - 520
         FPS  METRO   MOBA
  FPS    103      4      1
METRO      2    146      0
 MOBA      9      0    109
  - obsetvation
    - le meilleurs résultats est avec  :520 cluster, 93,9% d'accyracy, 0.11 en loss durant le test, ey 0.13 de loss durant le test
    - avec des variation vers 530, il n'y a aucune différence
    - je vais regarder si on peu baisser le nombre d cluster tout en gadant les même résultats



## Exp 20
- à la recherche de moisn de cluster
  - parametre
    - mouvement max : 0.0001
    - nb cluster : 5219
    - max loop : 1000 
    - gamma : 0.01
    - liste seed : [14, 1, 2, 12, 59, 96, 3, 11, 13, 18, 26, 30, 59] 
  - résultat
    - loss  : 519(0.11, 0.13)
    - accuracy :  519(0.938)
    - matrice 59
          FPS  METRO   MOBA
  FPS    102      5      1
METRO      6    142      0
 MOBA      9      0    109
  - obsetvation
    - le résultats d'avant avais plus d'accuracy, donc je vais le sélectionner le modèle à 520 cluster avec 0.01 gamma sous une seed de 3 (pour le moment)
    - utilisatio de l'aléatoire pour avoir une meilleur seed



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