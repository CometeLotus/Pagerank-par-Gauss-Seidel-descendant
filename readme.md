# README

## Description
Ce script implémente et compare deux algorithmes de calcul du PageRank : la méthode des puissances et la méthode de Gauss-Seidel, sur des graphes représentés sous forme de matrices creuses. Il mesure et trace le nombre d'itérations, le temps d'exécution et la consommation de mémoire pour chaque algorithme en fonction du facteur de damping alpha.

## Prérequis
Avant d'exécuter le script, assurez-vous que les packages Python suivants sont installés :

- `numpy`
- `scipy`
- `matplotlib`
- `tqdm`
- `tracemalloc`

Vous pouvez installer ces packages en utilisant pip :

```sh
pip install numpy scipy matplotlib tqdm tracemalloc
 ```
## Execution

Il suffit de taper dans le terminal 
```sh
 python PageRank.py  
 ``` 


## Détails

- Le script scanne le répertoire courant pour trouver des fichiers texte.
- Il affiche la liste des fichiers trouvés et demande à l'utilisateur de choisir un fichier.
- L'utilisateur choisit l'algorithme à utiliser (méthode des puissances, méthode de Gauss-Seidel ou les deux).
- L'utilisateur spécifie le nombre de simulations à exécuter.
- Le script lit et traite les données du fichier choisi.
- Les algorithmes de PageRank sont exécutés pour différentes valeurs de alpha.
- Les résultats sont affichés et tracés.