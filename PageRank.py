import os
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import tracemalloc
from tqdm import tqdm

# === Section 1: Fonctions utilitaires ===

def lire_texte(fichier):
    """
    Lit un fichier texte et retourne son contenu et le nombre de sommets.
    
    Args:
        fichier (str): Le chemin du fichier texte à lire.
        
    Returns:
        tuple: Un tuple contenant le contenu du fichier (list de str) et le nombre de sommets (int).
    """
    with open(fichier, 'r') as texte:
        contenu = texte.readlines()
    nb_sommets = int(contenu[0])
    return contenu, nb_sommets

def supprimer_redondances(lignes):
    """
    Supprime les redondances dans les lignes et retourne les nouvelles lignes, 
    le nombre de nœuds et le nombre total d'arêtes.
    
    Args:
        lignes (list of str): Les lignes du fichier texte.
        
    Returns:
        tuple: Un tuple contenant les nouvelles lignes (list of list of str), 
               le nombre de nœuds (int) et le nombre total d'arêtes (int).
    """
    noeuds = int(lignes[0])
    total_aretes = int(lignes[1])
    nouvelles_lignes = [line.split() for line in lignes[2:]]
    return nouvelles_lignes, noeuds, total_aretes

def convertir_str_en_num(nouvelles_lignes):
    """
    Convertit les chaînes de caractères en nombres dans les nouvelles lignes.
    
    Args:
        nouvelles_lignes (list of list of str): Les nouvelles lignes avec des chaînes de caractères.
        
    Returns:
        list of list of int/float: Les nouvelles lignes avec des nombres entiers et flottants.
    """
    for i in range(len(nouvelles_lignes)):
        nouvelles_lignes[i] = [int(nouvelles_lignes[i][j]) if j % 2 == 0 else float(nouvelles_lignes[i][j])
                               for j in range(len(nouvelles_lignes[i]))]
    return nouvelles_lignes

def vecteur_teleportation(lignes):
    """
    Crée un vecteur de téléportation basé sur les lignes fournies.
    
    Args:
        lignes (list of list of int/float): Les lignes de données.
        
    Returns:
        numpy.ndarray: Le vecteur de téléportation.
    """
    return np.array([0 if len(line) > 0 else 1 for line in lignes])

# === Section 2: Algorithmes de PageRank ===

def page_rank_sparse(alpha, matrice_csr, lignes_zero, tol=1e-6, max_iter=1000):
    """
    Calcule le PageRank en utilisant la méthode des puissances pour matrices creuses.
    
    Args:
        alpha (float): Le facteur de damping.
        matrice_csr (scipy.sparse.csr_matrix): La matrice creuse.
        lignes_zero (set): Les lignes avec des zéros.
        tol (float, optional): La tolérance pour la convergence. Default is 1e-6.
        max_iter (int, optional): Le nombre maximal d'itérations. Default is 1000.
        
    Returns:
        tuple: Un tuple contenant le vecteur PageRank (numpy.ndarray), 
               le nombre d'itérations (int) et le temps écoulé (float).
    """
    nb_sommets = matrice_csr.shape[0]
    rank = np.random.rand(nb_sommets)
    rank /= np.sum(rank)
    
    e = np.ones(nb_sommets)
    lignes_zero = np.array(list(lignes_zero), dtype=int)
    iteration = 0
    start_time = time.time()
    with tqdm(total=max_iter, desc="PageRank Puissances", leave=False) as pbar:
        while iteration < max_iter:
            m2 = alpha * matrice_csr @ rank
            m2 += (1 - alpha) * (e / nb_sommets)
            zero_sum_contrib = np.sum(rank[lignes_zero]) / nb_sommets
            m2 += zero_sum_contrib * e

            if np.linalg.norm(m2 - rank, 1) < tol:
                elapsed_time = time.time() - start_time 
                return m2 / np.sum(m2), iteration, elapsed_time

            rank = m2
            iteration += 1
            pbar.update(1)

    elapsed_time = time.time() - start_time 
    return rank / np.sum(rank), iteration, elapsed_time

def page_rank_gauss_seidel(alpha, matrice_csr, tol=1e-6, max_iter=1000):
    """
    Calcule le PageRank en utilisant la méthode de Gauss-Seidel pour matrices creuses.
    
    Args:
        alpha (float): Le facteur de damping.
        matrice_csr (scipy.sparse.csr_matrix): La matrice creuse.
        tol (float, optional): La tolérance pour la convergence. Default is 1e-6.
        max_iter (int, optional): Le nombre maximal d'itérations. Default is 1000.
        
    Returns:
        tuple: Un tuple contenant le vecteur PageRank (numpy.ndarray), 
               le nombre d'itérations (int) et le temps écoulé (float).
    """
    n = matrice_csr.shape[0]
    x = np.ones(n) / n
    teleportation = (1 - alpha) / n

    start_time = time.time()
    for iteration in tqdm(range(max_iter), desc="Gauss-Seidel Descendant", leave=False):
        x_new = np.copy(x)
        for i in range(n-1, -1, -1):  # Descendant
            row_start = matrice_csr.indptr[i]
            row_end = matrice_csr.indptr[i+1]
            sum_GS = matrice_csr.data[row_start:row_end] @ x_new[matrice_csr.indices[row_start:row_end]]
            x_new[i] = alpha * sum_GS + teleportation
        x_new /= np.sum(x_new)  # Renormalisation du vecteur
        if np.linalg.norm(x_new - x, 1) < tol:
            elapsed_time = time.time() - start_time
            return x_new, iteration + 1, elapsed_time
        x = x_new

    elapsed_time = time.time() - start_time
    return x, max_iter, elapsed_time

# === Section 3: Fonctions de mesure de performance ===

def mesurer_memoire_et_temps(func, *args):
    """
    Mesure la consommation de mémoire et le temps d'exécution d'une fonction.
    
    Args:
        func (callable): La fonction à mesurer.
        *args: Les arguments à passer à la fonction.
        
    Returns:
        tuple: Un tuple contenant le résultat de la fonction, 
               le temps écoulé (float) et la consommation de mémoire (float en MiB).
    """
    tracemalloc.start()
    start_time = time.time()
    result = func(*args)
    elapsed_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_usage = peak / 10**6  # Convert to MiB
    return result, elapsed_time, memory_usage

def imprimer_top_elements(vecteur_pr, n=5, methode="Méthode"):
    """
    Imprime les top N éléments les mieux classés dans le vecteur PageRank.
    
    Args:
        vecteur_pr (numpy.ndarray): Le vecteur PageRank.
        n (int, optional): Le nombre d'éléments à imprimer. Default is 5.
        methode (str, optional): Le nom de la méthode. Default is "Méthode".
    """
    top_indices = np.argsort(vecteur_pr)[-n:][::-1]
    print(f"\n\033[93mTop {n} éléments les mieux classés ({methode}) :\033[0m")
    for rank, index in enumerate(top_indices, 1):
        print(f"Rank {rank}: Element {index} avec un score de {vecteur_pr[index]:.6f}")

# === Section 4: Fonction principale ===

def main():
    """
    Fonction principale du script.
    Gère l'entrée de l'utilisateur, le traitement des données, 
    l'exécution des algorithmes de PageRank et l'affichage des résultats.
    """
    # Scanner le répertoire courant pour trouver les fichiers texte
    fichiers_txt = [f for f in os.listdir('.') if f.endswith('.txt')]
    if not fichiers_txt:
        print("Aucun fichier texte trouvé dans le répertoire courant.")
        return

    # Afficher la liste des fichiers texte trouvés
    print("Fichiers texte disponibles :")
    for i, fichier in enumerate(fichiers_txt):
        print(f"{i + 1}. {fichier}")

    # Demander à l'utilisateur de choisir un fichier
    choix = int(input("Entrez le numéro du fichier que vous souhaitez utiliser : ")) - 1
    if choix < 0 or choix >= len(fichiers_txt):
        print("Choix invalide.")
        return

    fichier = fichiers_txt[choix]

    # Demander à l'utilisateur pour quel algorithme tracer les courbes
    print("Choisissez l'algorithme pour lequel tracer les courbes :")
    print("1. Méthode des Puissances")
    print("2. Méthode de Gauss-Seidel")
    print("3. Les deux")
    choix_algo = int(input("Entrez le numéro correspondant à votre choix (1/2/3) : "))

    # Demander à l'utilisateur combien de simulations il souhaite exécuter
    nb_simulations = int(input("Entrez le nombre de simulations que vous souhaitez exécuter : "))

    # Demander à l'utilisateur s'il souhaite voir les informations additionnelles
    voir_infos_additionnelles = input("Souhaitez-vous voir les informations additionnelles (Top 5 final, ranking final, verification) ? (oui/non) : ").strip().lower() == 'oui'

    start_time_total = time.time()

    # === Section 5: Traitement des données ===

    tstart = time.time()
    lignes, lignes_totales = lire_texte(fichier)
    lignes, sommets, total_aretes = supprimer_redondances(lignes)
    lignes = convertir_str_en_num(lignes)
    tend = time.time()
    print(f"\033[92mTraitement des données terminé en {tend - tstart:.4f} secondes.\033[0m")

    print(f"\033[94mNombre de sommets : {sommets}\033[0m")
    print(f"\033[94mNombre d'arêtes : {total_aretes}\033[0m")

    # Création de la liste des arêtes
    tstart = time.time()
    liste_aretes = [ligne[2::2] for ligne in lignes]
    tend = time.time()
    print(f"\033[92mTemps nécessaire pour créer la liste des arêtes : {tend - tstart:.4f} secondes.\033[0m")

    # Création du vecteur de téléportation
    tstart = time.time()
    teleportation = vecteur_teleportation(liste_aretes)
    tend = time.time()
    print(f"\033[92mTemps pour créer le vecteur de téléportation : {tend - tstart:.4f} secondes.\033[0m")

    # Construction de la matrice creuse pour les algorithmes
    lignes, colonnes, donnees = [], [], []
    lignes_zero = set(range(sommets))
    for i, adj in enumerate(liste_aretes):
        if len(adj) > 0:
            lignes_zero.discard(i)
        for j in adj:
            lignes.append(j - 1)
            colonnes.append(i)
            donnees.append(1 / len(adj))
    matrice_csr = csr_matrix((donnees, (lignes, colonnes)), shape=(sommets, sommets))

    # === Section 6: Calcul du PageRank ===

    alphas = np.linspace(0.01, 0.99, nb_simulations)
    iterations_puissance_sparse = []
    iterations_gauss_seidel = []
    temps_puissance_sparse_list = []
    temps_gauss_seidel_list = []
    memoire_puissance_sparse = []
    memoire_gauss_seidel = []

    for alpha in tqdm(alphas, desc="Calcul du PageRank pour chaque alpha"):
        print(f"\n\033[96mAlpha : {alpha:.2f}\033[0m")
        
        if choix_algo in [1, 3]:
            # Méthode des puissances pour matrices creuses
            (vecteur_pr_sparse, iter_puissance_sparse, _), temps_puissance_sparse, mem_usage = mesurer_memoire_et_temps(page_rank_sparse, alpha, matrice_csr, lignes_zero)
            iterations_puissance_sparse.append(iter_puissance_sparse)
            temps_puissance_sparse_list.append(temps_puissance_sparse)
            memoire_puissance_sparse.append(mem_usage)
            print(f"\n\033[92mItérations Méthode des Puissances (Sparse) : {iter_puissance_sparse:<4d}, Temps : {temps_puissance_sparse:<7.4f} secondes, Mémoire : {mem_usage:<7.4f} MiB.\033[0m")
        
        if choix_algo in [2, 3]:
            # Méthode de Gauss-Seidel pour matrices creuses
            (vecteur_pr_gs, iter_gauss_seidel, _), temps_gauss_seidel, mem_usage = mesurer_memoire_et_temps(page_rank_gauss_seidel, alpha, matrice_csr, 1e-6, 1000)
            iterations_gauss_seidel.append(iter_gauss_seidel)
            temps_gauss_seidel_list.append(temps_gauss_seidel)
            memoire_gauss_seidel.append(mem_usage)
            print(f"\n\033[92mItérations Méthode de Gauss-Seidel : {iter_gauss_seidel:<4d}, Temps : {temps_gauss_seidel:<7.4f} secondes, Mémoire : {mem_usage:<7.4f} MiB.\033[0m")
        
        # Vérification de la somme des éléments du vecteur de PageRank
        if voir_infos_additionnelles and choix_algo in [1, 3]:
            somme_vecteur_pr_sparse = np.sum(vecteur_pr_sparse)
            print(f"\033[94mSomme des éléments du vecteur PageRank (Sparse) : {somme_vecteur_pr_sparse:.6f}\033[0m")
            if not np.isclose(somme_vecteur_pr_sparse, 1):
                print(f"\033[91mLa somme des éléments du vecteur PageRank (Sparse) n'est pas égale à 1.\033[0m")
        
        if voir_infos_additionnelles and choix_algo in [2, 3]:
            somme_vecteur_pr_gs = np.sum(vecteur_pr_gs)
            print(f"\033[94mSomme des éléments du vecteur PageRank (Gauss-Seidel) : {somme_vecteur_pr_gs:.6f}\033[0m")
            if not np.isclose(somme_vecteur_pr_gs, 1):
                print(f"\033[91mLa somme des éléments du vecteur PageRank (Gauss-Seidel) n'est pas égale à 1.\033[0m")

    end_time_total = time.time()
    total_elapsed_time = end_time_total - start_time_total
    print(f"\n\033[92mTemps total d'exécution du programme : {total_elapsed_time:.4f} secondes.\033[0m\n")

    # === Section 7: Affichage des résultats ===

    # Impression de la matrice finale de PageRank pour la dernière valeur d'alpha
    if voir_infos_additionnelles:
        if choix_algo in [1, 3]:
            print(f"\033[95mMatrice finale de PageRank (Méthode des Puissances - Sparse) :\033[0m")
            print(np.array2string(vecteur_pr_sparse, separator=', '))

        if choix_algo in [2, 3]:
            print(f"\n\033[95mMatrice finale de PageRank (Méthode de Gauss-Seidel) :\033[0m")
            print(np.array2string(vecteur_pr_gs, separator=', '))

    # Tracer le nombre d'itérations en fonction de alpha
    plt.figure(figsize=(10, 6))
    if choix_algo == 1:
        plt.plot(alphas, iterations_puissance_sparse, marker='o', label='Méthode des Puissances (Sparse) - Itérations')
    elif choix_algo == 2:
        plt.plot(alphas, iterations_gauss_seidel, marker='x', label='Méthode de Gauss-Seidel - Itérations', color='orange')
    else:
        plt.plot(alphas, iterations_puissance_sparse, marker='o', label='Méthode des Puissances (Sparse) - Itérations')
        plt.plot(alphas, iterations_gauss_seidel, marker='x', label='Méthode de Gauss-Seidel - Itérations')

    plt.xlabel('Facteur de damping (alpha)')
    plt.ylabel('Nombre d\'itérations')
    plt.title('Nombre d\'itérations avant convergence en fonction du facteur de damping (alpha)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Tracer le temps d'exécution en fonction de alpha
    plt.figure(figsize=(10, 6))
    if choix_algo == 1:
        plt.plot(alphas, temps_puissance_sparse_list, marker='o', label='Méthode des Puissances (Sparse) - Temps')
    elif choix_algo == 2:
        plt.plot(alphas, temps_gauss_seidel_list, marker='x', label='Méthode de Gauss-Seidel - Temps', color='orange')
    else:
        plt.plot(alphas, temps_puissance_sparse_list, marker='o', label='Méthode des Puissances (Sparse) - Temps')
        plt.plot(alphas, temps_gauss_seidel_list, marker='x', label='Méthode de Gauss-Seidel - Temps')

    plt.xlabel('Facteur de damping (alpha)')
    plt.ylabel('Temps d\'exécution (secondes)')
    plt.title('Temps d\'exécution en fonction du facteur de damping (alpha)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Tracer la consommation de mémoire en fonction de alpha
    plt.figure(figsize=(10, 6))
    if choix_algo == 1:
        plt.plot(alphas, memoire_puissance_sparse, marker='o', label='Méthode des Puissances (Sparse) - Mémoire')
    elif choix_algo == 2:
        plt.plot(alphas, memoire_gauss_seidel, marker='x', label='Méthode de Gauss-Seidel - Mémoire', color='orange')
    else:
        plt.plot(alphas, memoire_puissance_sparse, marker='o', label='Méthode des Puissances (Sparse) - Mémoire')
        plt.plot(alphas, memoire_gauss_seidel, marker='x', label='Méthode de Gauss-Seidel - Mémoire')

    plt.xlabel('Facteur de damping (alpha)')
    plt.ylabel('Consommation de mémoire (MiB)')
    plt.title('Consommation de mémoire en fonction du facteur de damping (alpha)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Imprimer le top 5 des éléments les mieux classés
    if voir_infos_additionnelles:
        if choix_algo in [1, 3]:
            imprimer_top_elements(vecteur_pr_sparse, 5, "Méthode des Puissances (Sparse)")
        if choix_algo in [2, 3]:
            imprimer_top_elements(vecteur_pr_gs, 5, "Méthode de Gauss-Seidel")

if __name__ == "__main__":
    main()
