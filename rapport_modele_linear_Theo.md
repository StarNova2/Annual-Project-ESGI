jour 1 : 
décryptage du code de classification Linéaire 
compréhension de la méthode de la génération de données, de l'affichage, du découpage 

jour 2 : (plusieurs nuit ce sont passé entre les jours)
compréhension de l'entrainement :
on pioche aléatoirement une données
on créer un tuples de 3 avec 1 puis les position de la données
on prend le label de la données pour savoir si c'est bon ou pas(1 ou -1) 
on utilise un produit scalaire pour ressortir un 1 ou -1
on met à jour W selon les résultats obtenu auparavant 

jour 3 : 
Meilleur compréhension de l'entrainement : 
le 1 du tuple de 3 sert de vérification 
la fonction linéaire fait un produit scalaire pour savoir quel est l'état de w selon la donnée
mise à jour de w --> si bon label et bon positionnement --> aucun changement, sinon mise à jour selon un pas d'apprentissage et le tuple de la données actuel

jour 4 : 
semblant de compréhension de w [x :f32, y : f32, z : f32] --> une droite de y à z qui débute à x hauteur 

jour 5 (et plus)
écriture du programme en python et de la lib rust 
problème rencontrer : mal compréhension de ctype	--> déclarer à chaque fois les return des fonctions
							--> déclarer des variables en entrée 
							--> précision des valeur avec dtype (float, int)
les fonctions de la lib n'accepte que les structures "simple" (donc pas les tuples)


jour 6 : 
continuation du code ! :
au lieu d'utiliser une lib : produit scalaire "simple" : weights1*xinput1 + weights2*xinput2 + weights3*xinput3
complication de la fonction d'entrainement --> pas de tuples à l'entrée mais utilisation d'une structure avec les 3 point [x, y, z]
ça marche ! 

jour 7: 
Notebook cas de test : 
complétion du code utiliser auparavant pour faire les cas de test
mal compréhension de l'ensemble des données, des test à effectuer
modification du code données avec les dtypes
modification de la lib pour ajouter le pas d'apprentissage lors de l'entrainement
mise en place d'un trait noir pour voir la séparation

réussite du cas de test linear simple

jour 8 : 
tes du reste du Notebook : 
Linear Multiple : entrainement ne marche pas, les valeurs n'arrête pas d'évoluer 
		--> changement du pas d'apprentissage
		--> changement de la librairie sur l'apprentissage
		le trait noir semble être bonne, mais l'axe ait mauvais 
XOR :
test sans suite

Muti Linear 3 classes : 
test sans suite 