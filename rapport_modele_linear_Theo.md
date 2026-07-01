### jour 1 : 
décryptage du code de classification Linéaire 
compréhension de la méthode de la génération de données, de l'affichage, du découpage 

### jour 2 : (plusieurs nuit ce sont passé entre les jours)
compréhension de l'entrainement :
on pioche aléatoirement une données
on créer un tuples de 3 avec 1 puis les position de la données
on prend le label de la données pour savoir si c'est bon ou pas(1 ou -1) 
on utilise un produit scalaire pour ressortir un 1 ou -1
on met à jour W selon les résultats obtenu auparavant 

### jour 3 : 
Meilleur compréhension de l'entrainement : 
le 1 du tuple de 3 sert de vérification 
la fonction linéaire fait un produit scalaire pour savoir quel est l'état de w selon la donnée
mise à jour de w --> si bon label et bon positionnement --> aucun changement, sinon mise à jour selon un pas d'apprentissage et le tuple de la données actuel

### jour 4 : 
semblant de compréhension de w [x :f32, y : f32, z : f32] --> une droite de y à z qui débute à x hauteur 

### jour 5 (et plus)
écriture du programme en python et de la lib rust 
problème rencontrer : mal compréhension de ctype	--> déclarer à chaque fois les return des fonctions
							--> déclarer des variables en entrée 
							--> précision des valeur avec dtype (float, int)
les fonctions de la lib n'accepte que les structures "simple" (donc pas les tuples)


### jour 6 : 
continuation du code ! :
au lieu d'utiliser une lib : produit scalaire "simple" : weights1*xinput1 + weights2*xinput2 + weights3*xinput3
complication de la fonction d'entrainement --> pas de tuples à l'entrée mais utilisation d'une structure avec les 3 point [x, y, z]
ça marche ! 

### jour 7: 
Notebook cas de test : 
complétion du code utiliser auparavant pour faire les cas de test
mal compréhension de l'ensemble des données, des test à effectuer
modification du code données avec les dtypes
modification de la lib pour ajouter le pas d'apprentissage lors de l'entrainement
mise en place d'un trait noir pour voir la séparation

réussite du cas de test linear simple

### jour 8 : 
tes du reste du Notebook : 
Linear Multiple : entrainement ne marche pas, les valeurs n'arrête pas d'évoluer 
		--> changement du pas d'apprentissage
		--> changement de la librairie sur l'apprentissage
		le trait noir semble être bonne, mais l'axe ait mauvais 
XOR :
test sans suite

Muti Linear 3 classes : 
test sans suite 


### jour 9 (17/05/2026) :
résolution du bug concernant le **Linear Multiple**. Pour rappel, le trait créer semblais décalé à l'horizontal par rapport au données.
Pour résoudre le problème, j'ai d'abord différencier **Linear Simple** avec mon code, cependant ça semblais correspondre. 
Donc je me suis dit que le problème venais de la création des données

Les données semblait correct (liste de liste contenant 2 valeurs). Mais les données donnant le résultats (1 ou -1) est en liste de liste d'une valeur. Alors que du coté du linear simplex, c'est une liste de valeur.
L'entrainement regadrais donc une liste de liste au lieu d'une simple liste. 

Après avoir corrigé les données entrantes, le code s'est mis à fonctionner (avec des valeur d'entrainement et de pas d'apprentissage plus réalisable que mes précédent test).
Je réaliserais des tests sur le pas d'entrainement et sur le nombre de tour plus tard

## jour 10 (24/05/2026) : 
- Mise en place du **Linear Simple** sur le cas **XOR** : 
  - Dans la théorie, une droite ne pourrais pas faire la classification d'un xor. 
  - Dans la pratique, on vois bien sur le graphique que la ligne n'arrive jamais à correctement classifier
  
- Mise en place du **Linear Simple** sur le cas **Cross** :
  - Dans la théorie, une droite ne suffis pas à séparer les 2 classes
  - Dans la pratique, quand ça fonctionne bien (bug expliqé après) la droite coupe à la vertical le jeu de données, mais n'arrive séparer les 2 classes.
  - Bug remarqué, lors de mes tests, 2 fois sur 3 je n'avais pas de résultats (pas = 0.01, boucle = 2000 (oui j'y allais un peu fort)). J'ai décidé d'afficher les poids et j'ai remarqué que le poid w.b équivalais à "nan".
Après avoir redémarer mon pc, je n'arrive plus à reproduire l'erreur. Peut être que c'était dû à une surcharge matériel?

- Mise en place du **Linear Simple** sur le cas **Multi Linear 3 classes** :
  - Dans la théorie, une droite ne suffirais pas à classifier 3 classes. 2 droite pourrais fonctionner avec -1 (pour la classes du dessous) correspondant à une classes, et 1 de la 1 ère droite & 1 de la 2 ème droite équi vaut à la 3 ème classes.
En regardant mieux le graphique, j'ai vu qu'une droite à l'horizontale pourrais rerésenter la classes rouge, donc 3 droite pourrais "facilement" représenter les 3 classes.
  - Dans la pratiques, après avoir implémenter l'entrainement sur 3 droite avec toutes les données, j'ai remarqué que l'entrainement ne marchais pas. Il faudrais surrement donné à chaque droite seulement 2 classes (bleu et les autres, rouge et les autres ou vert et les autres).

  - Bug remarquer, retour du "nan". Après des recherche, j'ai compris que "nan" signifiait "Not a Number". Nan intervient quand le float deviens bien trop grand, donc pour limiter son apparttion je dois faire évoluer les poids d'un façons plus douce en faisant moins de boucle, ou en réduisant le pas d'apprentissage.

  