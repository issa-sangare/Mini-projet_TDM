#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import re


# In[2]:


path = 'eren_rgb.jpg'
path1 = 'text1.txt'
path2 = 'reconstiued1.jpg'
num_rows = 2
num_colors = 16
num_blocs = 16
image = np.array(Image.open(path))


# # (-) $Caractéristiques$ $de$ $l'image$

# In[3]:


def details(path):
    img = Image.open(path)
    size = img.size
    format = img.format
    mode = img.mode
    resolution = img.info.get('dpi')
    definition = size[0] * size[1]
    definition2 = size
    if resolution is not None:
        size = (size[0]/ resolution[0], size[1] / resolution[1])
    else:
        size = (0, 0)  # or any default resolution you want to use
    
    poids = os.path.getsize(path) / (1000.0)  # Convert to kilobytes
    
    if mode == 'RGB':
        trsc = definition * 3 * img.bits
    else:
        trsc = definition * 1 * img.bits
    
    trsc /= (1024 * 8)
    
    taux_compression = 100 * (1 - (poids/trsc))
        
    return size, format, mode, resolution, definition, definition2, poids, trsc, taux_compression


# In[4]:


# Exemple d'utilisation
result = details(path)
print("Size (inch):", result[0])
print("Format:", result[1])
print("Mode:", result[2])
print("Resolution:", result[3])
print("Definition (pixels):", result[4])
print("Définition (L * C):", result[5])
print("Taille en mémoire avec compression:", result[6], "kB")
print("Taille réelle sans compression:", result[7], "kB")
print("Taux de compression: ", result[8], "%")


# In[5]:


def display_image_with_palette(image, color_palette, palette_indices, num_rows):
    num_colors = len(color_palette)
    num_cols = num_colors // num_rows

    image_with_palette = color_palette[palette_indices]

    plt.figure(figsize=(30, 15))
    for i, color in enumerate(color_palette):
        plt.subplot(num_rows, num_cols, i + 1)
        color_patch = np.zeros((100, 100, 3), dtype=np.uint8)
        color_patch[:, :] = color
        plt.imshow(color_patch)
        plt.title(f'Couleur {i+1}')
        plt.axis('off')
    plt.suptitle('Palette de couleurs', fontsize=16)
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(30, 15))
    fig.subplots_adjust(wspace=0.01)

    axs[0].imshow(image)
    axs[0].set_title('Image originale')
    axs[0].axis('off')

    axs[1].imshow(image_with_palette)
    axs[1].set_title(f'Image avec la palette de {num_colors} couleurs')
    axs[1].axis('off')

    fig.tight_layout()

    plt.show()


# In[6]:


def plot_images(reconstructed_image, image1, num_colors):
    fig, axs = plt.subplots(1, 2, figsize=(30, 15))
    fig.subplots_adjust(wspace=0.01)

    axs[0].imshow(reconstructed_image)
    axs[0].set_title('Image reconstruite')
    axs[0].axis('off')

    axs[1].imshow(image1)
    axs[1].set_title(f'Image avec la palette de {num_colors} couleurs (median-cut)')
    axs[1].axis('off')

    fig.tight_layout()

    plt.show()


# In[7]:


def MSE(image_d_origine, image_traitee):
    # Get the dimensions of the images
    L, C = np.array(image_d_origine).shape[:2]  # Assuming it's a 2D image, getting rows and columns
    image_traitee = np.array(image_traitee).reshape(np.array(image_d_origine).shape)
    # Convert the images to NumPy arrays
    origine = np.array(image_d_origine)
    traitee = np.array(image_traitee)

    # Calculate the sum of squared differences
    somme_differences_carrees = np.sum((origine - traitee) ** 2)

    # Calculate (1 / (L * C)) * somme
    resultat = (1 / (L * C)) * somme_differences_carrees

    return resultat


# In[8]:


def taux_compression(img_path, compressed_file_path):
    img = Image.open(img_path)
    
    if img.mode == 'RGB':
        pixel_size = 3  # Nombre de canaux de couleur (RGB)
    else:
        pixel_size = 1  # Pour d'autres modes de couleur
    
    if img.bits == 8:  # Taille en bits par pixel
        pixel_bits = 8
    else:
        pixel_bits = 16  # Peut être ajusté selon les besoins
    
    img_data_size = img.size[0] * img.size[1] * pixel_size * pixel_bits  # Taille en bits de l'image
    
    # Lecture des données compressées à partir du fichier .irm
    with open(compressed_file_path, 'rb') as f:
        compressed_data = f.read()
    
    compressed_data_size = len(compressed_data) * 8  # Taille des données compressées en bits
    
    compression_ratio = img_data_size / compressed_data_size  # Ratio de compression
    
    return compression_ratio


# In[9]:


def rgb_to_ycrcb(rgb_image):
    rgb_image = np.array(rgb_image)
    ycrcb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)
    return ycrcb_image


# In[10]:


def ycrcb_to_rgb(ycrcb_image):
    rgb_image = cv2.cvtColor(np.array(ycrcb_image), cv2.COLOR_YCrCb2RGB)
    # Convert numpy array to PIL Image
    rgb_image_pil = Image.fromarray(rgb_image)
    # Save the PIL Image
    return rgb_image_pil


# ### (c). $Application$

# In[11]:


ycrcb_image = rgb_to_ycrcb(image)

plt.figure(figsize=(30, 15))  # Définit la taille de la figure à 10 pouces de largeur et 8 pouces de hauteur
plt.imshow(ycrcb_image)  # Affiche l'image Y dans les tons de gris
plt.xticks([])  # Supprime les marques sur l'axe des x
plt.yticks([])  # Supprime les marques sur l'axe des y
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Ajuste les marges pour enlever l'espace entre l'image et les bords du plot
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(30, 15))
fig.subplots_adjust(wspace=0.01)

axs[0].imshow(ycrcb_image[:,:,0], cmap='gray')
axs[0].axis('off')

axs[1].imshow(ycrcb_image[:,:,1], cmap='gray')
axs[1].axis('off')

axs[2].imshow(ycrcb_image[:,:,2], cmap='gray')
axs[2].axis('off')

fig.tight_layout()

plt.show()

rgb_image = ycrcb_to_rgb(ycrcb_image)

plt.figure(figsize=(30, 15))  # Définit la taille de la figure à 10 pouces de largeur et 8 pouces de hauteur
plt.imshow(rgb_image)  # Affiche l'image Y dans les tons de gris
plt.xticks([])  # Supprime les marques sur l'axe des x
plt.yticks([])  # Supprime les marques sur l'axe des y
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Ajuste les marges pour enlever l'espace entre l'image et les bords du plot
plt.show()

# Convert the RGB image to numpy array
rgb_array = np.array(rgb_image)

fig, axs = plt.subplots(1, 3, figsize=(30, 15))
fig.subplots_adjust(wspace=0.01)

axs[0].imshow(rgb_array[:,:,0], cmap='gray')
axs[0].axis('off')

axs[1].imshow(rgb_array[:,:,1], cmap='gray')
axs[1].axis('off')

axs[2].imshow(rgb_array[:,:,2], cmap='gray')
axs[2].axis('off')

fig.tight_layout()

plt.show()


# In[12]:


mse_value = MSE(image, rgb_image)
print("MSE entre l'image originale et l'image reconstruite :", mse_value)
plot_images(image, rgb_image, num_colors)


# # (1). $Création$ $de$ $la$ $palette$ $de$ $couleurs$

# ### (a). $Sens$ $normal$

# In[13]:


def kmeans_clustering_palette(image, num_colors):
    # Convertir l'image en un tableau 2D de pixels
    pixels = np.reshape(image, (-1, 3))  # (nombre de pixels, 3 canaux de couleur)

    # Appliquer l'algorithme de k-means clustering
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    # Obtenir les centres des clusters (couleurs dominantes)
    color_palette = kmeans.cluster_centers_.astype(int)

    # Associer chaque pixel à l'indice de couleur dans la palette
    labels = kmeans.predict(pixels)

    # Reformater les indices des couleurs dans la palette selon la forme de l'image originale
    palette_indices = np.reshape(labels, np.array(image).shape[:2])

    return color_palette, palette_indices


# In[14]:


def median_cut_palette(image, num_colors):
    # Convertir l'image en un tableau 2D de pixels
    pixels = np.reshape(image, (-1, 3))

    # Initialiser la liste des cubes de couleur avec le cube contenant tous les pixels
    cubes = [pixels]

    # Répéter jusqu'à ce que le nombre de cubes atteigne le nombre de couleurs souhaité
    while len(cubes) < num_colors:
        # Sélectionner le cube le plus grand
        largest_cube_index = np.argmax([cube.shape[0] for cube in cubes])
        largest_cube = cubes.pop(largest_cube_index)

        # Trouver l'axe dominant pour diviser le cube
        axis = np.argmax(np.max(largest_cube, axis=0) - np.min(largest_cube, axis=0))

        # Trier les pixels du cube le long de l'axe dominant
        sorted_cube = largest_cube[largest_cube[:, axis].argsort()]

        # Diviser le cube en deux parties égales
        split_index = len(sorted_cube) // 2
        cube1 = sorted_cube[:split_index]
        cube2 = sorted_cube[split_index:]

        # Ajouter les deux nouveaux cubes à la liste
        cubes.append(cube1)
        cubes.append(cube2)

    # Calculer les couleurs moyennes pour chaque cube
    color_palette = [np.mean(cube, axis=0) for cube in cubes]
    color_palette = np.array(color_palette).astype(np.uint8)

    # Calculer les indices des couleurs dans la palette pour chaque pixel de l'image
    palette_indices = np.zeros(len(pixels), dtype=int)
    for i, pixel in enumerate(pixels):
        distances = np.linalg.norm(color_palette - pixel, axis=1)
        palette_indices[i] = np.argmin(distances)

    palette_indices = np.reshape(palette_indices, np.array(image).shape[:2])

    return color_palette, palette_indices


# In[15]:


def octree_palette(image, num_colors):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    img_rgb = image.convert("RGB")
    img_with_palette = img_rgb.quantize(colors=num_colors, method=Image.FASTOCTREE)
    palette = img_with_palette.getpalette()[:num_colors * 3]
    palette = np.array(palette).reshape(-1, 3)
    indices = np.array(img_with_palette)

    # Retourner la palette de couleurs et leurs indices
    return palette, indices


# ### (b). $Sens$ $inverse$

# In[16]:


def inverse_kmeans_clustering_palette(color_palette, palette_indices):
    # Récupérer les dimensions de l'image à partir des indices de palette
    height, width = palette_indices.shape

    # Initialiser une image vide avec les dimensions récupérées
    reconstructed_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Remplir l'image reconstruite avec les couleurs de la palette
    for i in range(height):
        for j in range(width):
            color_index = palette_indices[i, j]
            reconstructed_image[i, j] = color_palette[color_index]

    return reconstructed_image


# In[17]:


def inverse_median_cut_palette(color_palette, palette_indices):
    # Récupérer les dimensions de l'image à partir des indices de palette
    height, width = palette_indices.shape

    # Initialiser une image vide avec les dimensions récupérées
    reconstructed_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Remplir l'image reconstruite avec les couleurs de la palette
    for i in range(height):
        for j in range(width):
            color_index = palette_indices[i, j]
            reconstructed_image[i, j] = color_palette[color_index]
  
    return reconstructed_image


# In[18]:


def inverse_octree_palette(color_palette, palette_indices):
    color_palette = color_palette.reshape(-1)
    color_palette = [(int(color_palette[i]), int(color_palette[i + 1]), int(color_palette[i + 2])) for i in range(0, len(color_palette), 3)]
    reconstructed_image = Image.new("RGB", (palette_indices.shape[1], palette_indices.shape[0]))
    reconstructed_image.putdata([color_palette[idx] for idx in palette_indices.flatten()])
  
    return reconstructed_image


# ### (c). $Application$

# In[19]:


# Clustering des couleurs de l'image
color_palette1, palette_indices1 = kmeans_clustering_palette(ycrcb_image, num_colors)

# Reconstruction de l'image à partir de la palette de couleurs
reconstructed_image1 = inverse_kmeans_clustering_palette(color_palette1, palette_indices1)

# Afficher l'image originale avec la palette de couleurs
display_image_with_palette(ycrcb_image, color_palette1, palette_indices1, num_rows)

# Calculer la MSE entre l'image originale et l'image reconstruite
mse_value1 = MSE(ycrcb_image, reconstructed_image1)
print("MSE entre l'image originale et l'image reconstruite 1:", mse_value1)


# In[20]:


# Afficher l'image reconstruite et l'image originale avec la palette de couleurs
plot_images(reconstructed_image1, ycrcb_image, num_colors)


# # (2). $Mappage$ $des$ $pixels$ $de$ $l'image$

# ### (a). $Sens$ $normal$

# In[21]:


def map_to_palette(image, color_palette):
    # Assurez-vous que l'image est sous forme de tableau numpy
    image = np.array(image)
    
    # Redimensionner les pixels en une matrice (nombre de pixels, 3 canaux de couleur)
    pixels = np.reshape(image, (-1, 3))  

    # Calculer la distance de chaque pixel à chaque couleur de la palette
    distances = np.linalg.norm(pixels[:, np.newaxis] - color_palette, axis=2)

    # Obtenir l'indice de la couleur la plus proche pour chaque pixel
    indices = np.argmin(distances, axis=1)

    # Reformater les indices des couleurs selon la forme de l'image originale
    mapped_indices = np.reshape(indices, image.shape[:2])

    return np.array(mapped_indices)


# ### (b). $Sens$ $inverse$

# In[22]:


def inverse_map_to_palette(mapped_indices, color_palette):
    # Obtenir les dimensions de l'image reconstruite
    height, width = mapped_indices.shape

    # Créer une image vide
    reconstructed_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Pour chaque pixel, assigner la couleur correspondante de la palette
    for i in range(height):
        for j in range(width):
            color_index = mapped_indices[i, j]
            reconstructed_image[i, j] = color_palette[color_index]

    return Image.fromarray(reconstructed_image)


# ### (c). $Application$

# In[23]:


# Application de la fonction map_to_palette
mapped_indices1 = map_to_palette(image, color_palette1)

# Reconstruction de l'image à partir des indices mappés
reconstructed_image_from_mapping1 = inverse_map_to_palette(mapped_indices1, color_palette1)

# Calcul de la MSE entre l'image originale et l'image reconstruite à partir des indices mappés
mse_mapped1 = MSE(np.array(color_palette1[palette_indices1]), reconstructed_image_from_mapping1)

# Afficher la valeur de la MSE
print("MSE entre l'image originale et l'image reconstruite 1 à partir des indices mappés:", mse_mapped1)


# In[24]:


def subdivision(matrice, taille_blocs):
    # Dimensions de la matrice initiale
    lignes, colonnes = matrice.shape
    
    # Dimensions des blocs
    blocs_lignes, blocs_colonnes = (taille_blocs, taille_blocs)
    
    # Calcul des dimensions des blocs avec zéros ajoutés
    new_blocs_lignes = (lignes + blocs_lignes - 1) // blocs_lignes
    new_blocs_colonnes = (colonnes + blocs_colonnes - 1) // blocs_colonnes
    
    # Initialisation de la matrice des sous-matrices
    sous_matrices = np.zeros((new_blocs_lignes, new_blocs_colonnes, blocs_lignes, blocs_colonnes))
    
    # Remplissage de la matrice des sous-matrices
    for i in range(new_blocs_lignes):
        for j in range(new_blocs_colonnes):
            sous_matrices[i, j, :min(blocs_lignes, lignes - i*blocs_lignes), :min(blocs_colonnes, colonnes - j*blocs_colonnes)] = \
                matrice[i*blocs_lignes:(i+1)*blocs_lignes, j*blocs_colonnes:(j+1)*blocs_colonnes]
    
    # Convertir la matrice de sous-matrices en une liste de matrices 2D
    liste_matrices = []
    for i in range(new_blocs_lignes):
        for j in range(new_blocs_colonnes):
            liste_matrices.append(sous_matrices[i, j])
    
    return liste_matrices, (lignes, colonnes)


# In[25]:


def reconstitution(liste_matrices, dimensions):# Récupération des dimensions de la matrice initiale
    lignes, colonnes = dimensions
    
    # Initialisation de la matrice résultante avec des zéros
    matrice_resultante = np.zeros((lignes, colonnes))
    
    # Dimensions des blocs
    taille_blocs = liste_matrices[0].shape
    
    # Nombre de blocs
    nb_blocs_lignes = (lignes + taille_blocs[0] - 1) // taille_blocs[0]
    nb_blocs_colonnes = (colonnes + taille_blocs[1] - 1) // taille_blocs[1]
    
    # Recombinaison des sous-matrices dans la matrice résultante
    for i in range(nb_blocs_lignes):
        for j in range(nb_blocs_colonnes):
            matrice_resultante[i*taille_blocs[0]:(i+1)*taille_blocs[0], j*taille_blocs[1]:(j+1)*taille_blocs[1]] = \
                liste_matrices[i*nb_blocs_colonnes + j][:min(taille_blocs[0], lignes - i*taille_blocs[0]), :min(taille_blocs[1], colonnes - j*taille_blocs[1])]
    
    return np.array(matrice_resultante, dtype=np.int64)


# In[26]:


# Appliquer la fonction de subdivision
liste_blocs, dimensions_originales = subdivision(mapped_indices1, num_blocs)

# Afficher les dimensions de la matrice initiale
print("Dimensions de la matrice initiale :", dimensions_originales)

# Afficher les blocs obtenus après subdivision
print("Blocs obtenus après subdivision :")
for bloc in liste_blocs:
    print(bloc, "\n")

# Appliquer la fonction de reconstitution
matrice_reconstituee = reconstitution(liste_blocs, dimensions_originales)

# Calculer la MSE entre la matrice d'origine et la matrice reconstituée
mse_value = MSE(mapped_indices1, matrice_reconstituee)
print("MSE entre la matrice d'origine et la matrice reconstituée :", mse_value)

# Comparer les matrices d'origine et reconstituée directement
comparaison = np.array_equal(mapped_indices1, matrice_reconstituee)
print("Les matrices d'origine et reconstituée sont identiques :", comparaison)


# In[27]:


def sous_echantillonnage_4_2_0(image):
    # Sous-échantillonnage de l'image
    sub_image = image[::2, ::2]
    
    return sub_image


# In[28]:


def sous_echantillonnage_4_2_2(image):
    # Sous-échantillonnage de l'image
    sub_image = image[::1, ::2]
    
    return sub_image


# In[29]:


def sous_echantillonnage_4_4_4(image):
    # La fonction ne fait rien car il n'y a pas de sous-échantillonnage pour le 4:4:4
    return image.copy() 


# In[30]:


def inverse_sous_echantillonnage_4_2_0(imSub):
    # Répétition des pixels pour restaurer la résolution horizontale
    H, W = imSub.shape
    res_h = np.repeat(imSub, 2, axis=1)
    
    # Répétition des pixels pour restaurer la résolution verticale
    res_v = np.repeat(res_h, 2, axis=0)
    
    return res_v


# In[31]:


def inverse_sous_echantillonnage_4_2_2(imSub):
    # Répétition des colonnes pour restaurer les canaux de chrominance
    res_h = np.repeat(imSub, 2, axis=1)
    
    return res_h


# In[32]:


def inverse_sous_echantillonnage_4_4_4(imSub):
    # L'inverse du sous-échantillonnage 4:4:4 est simplement une copie de l'image d'entrée
    return imSub.copy()


# In[33]:


# Sous-échantillonnage de l'image
sub_image1 = sous_echantillonnage_4_2_0(liste_blocs[1])

# Répétition des pixels pour restaurer l'image à sa résolution d'origine
reconstituted_image1 = inverse_sous_echantillonnage_4_2_0(sub_image1)

# Calculer la MSE entre l'image d'origine et l'image reconstituée
mse_reconstituted1 = MSE(liste_blocs[1], reconstituted_image1)
print("MSE entre l'image d'origine et l'image reconstituée 1 :", mse_reconstituted1)


# In[34]:


def ligne_scan(matrice):
  shape = matrice.shape
  vecteur = matrice.flatten()
  return np.array(vecteur), shape


# In[35]:


def colonne_scan(matrice):
    shape = matrice.shape
    vecteur = np.ravel(matrice, order = 'F')
    return np.array(vecteur), shape


# In[36]:


def zigzag_scan(matrice):
  shape = matrice.shape
  vecteur = np.concatenate([np.diagonal(matrice[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1 - matrice.shape[0], matrice.shape[0])])
  return np.array(vecteur), shape


# In[37]:


def inverse_ligne_scan(vecteur, shape):
  mat = np.array(vecteur).reshape((shape[0], shape[1]))
  return mat


# In[38]:


def inverse_colonne_scan(vecteur, shape):
    mat = np.array(vecteur).reshape((shape[0], shape[1]))
    return mat.T


# In[39]:


def inverse_zigzag_scan(vector, shape):
    rows, cols = shape
    mat = np.zeros((rows, cols))
    row, col = 0, 0
    direction = 1

    for i in range(rows * cols):
        mat[row][col] = vector[i]
        if direction == 1:
            if col == cols - 1:
                row += 1
                direction = -1
            elif row == 0:
                col += 1
                direction = -1
            else:
                row -= 1
                col += 1
        else:
            if row == rows - 1:
                col += 1
                direction = 1
            elif col == 0:
                row += 1
                direction = 1
            else:
                row += 1
                col -= 1

    return mat


# In[40]:


# Compression de l'image
vecteur_comprime1, shape = ligne_scan(sub_image1)

# Décompression de l'image
vecteur_reconst1 = inverse_ligne_scan(vecteur_comprime1, shape)

# Calcul du MSE entre l'image originale et celle reconstruite
mse_value1 = MSE(sub_image1, vecteur_reconst1)
print("MSE entre la matrice d'origine et la matrice reconstituée :", mse_value1)

# Comparer les matrices d'origine et reconstituée directement
comparaison1 = np.array_equal(sub_image1, vecteur_reconst1)
print("Les matrices d'origine et reconstituée sont identiques :", comparaison1)


# In[64]:


def rle_encode(data):
    encoded_data = ""
    current_char = data[0]
    count = 1

    for i in range(1, len(data)):
        if data[i] == current_char:
            count += 1
        else:
            if count > 1:  # Ajout de cette condition
                encoded_data += str(current_char) + "_" + str(count) + "*"
            else:
                encoded_data += str(current_char) + "*"
            current_char = data[i]
            count = 1

    if count > 1:  # Ajout de cette condition
        encoded_data += str(current_char) + "_" + str(count)
    else:
        encoded_data += str(current_char)

    return encoded_data


# In[65]:


def rle_decode(encoded_data):
    decoded_data = []
    encoded_data_split = encoded_data.split('*')

    for item in encoded_data_split:
        if "_" in item:
            char, count = item.split("_")
            char = float(char)
            if char.is_integer():
                char = int(char)
            decoded_data.extend([char] * int(count))
        else:
            char = float(item)
            if char.is_integer():
                char = int(char)
            decoded_data.append(char)

    return decoded_data


# In[66]:


# Exemple d'utilisation
data1 = vecteur_comprime1
encoded_data1 = rle_encode(data1)
print(encoded_data1)

decoded_data1 = rle_decode(encoded_data1)
print(decoded_data1)
print("Départ =? Arrivée: ", (data1 == decoded_data1).all())


# In[67]:


print(vecteur_comprime1.dtype)
print(data1.dtype)


# In[68]:


def ext_chr(chaine):
    occ = {c: chaine.count(c) for c in set(chaine)}
    return sorted(occ.items(), key=lambda x: x[1], reverse=True)

# Fonction principale implémentant le codage de Huffman
def huffman_dictionnary_code(node, binString=''):
    if isinstance(node, str):
        return {node: binString}
    (l, r) = node
    d = {}
    d.update(huffman_dictionnary_code(l, binString + '0'))
    d.update(huffman_dictionnary_code(r, binString + '1'))
    return d

# Construction de l'arbre de Huffman
def huffman_tree(chaine):
    nodes = ext_chr(chaine)
    
    while len(nodes) > 1:
        (key1, c1) = nodes.pop()
        (key2, c2) = nodes.pop()
        node = (key1, key2)
        nodes.append((node, c1 + c2))
        nodes.sort(key=lambda x: x[1], reverse=True)
        
    return nodes[0][0]


# In[69]:


def encoding_huffman(string):
    nodes = huffman_tree(string)
    huffmanCode = huffman_dictionnary_code(nodes)
    
    compressed_string = ''
    for char in string:
        compressed_string += huffmanCode[char]
    return compressed_string, nodes


# In[70]:


def decoding_huffman(compressed_string, huffman_tree):
    decoded_string = ''
    current_node = huffman_tree
    
    for bit in compressed_string:
        if bit == '0':
            current_node = current_node[0]
        else:
            current_node = current_node[1]
        
        if isinstance(current_node, str):
            decoded_string += current_node
            current_node = huffman_tree
    
    return decoded_string


# In[71]:


compressed_text, tree = encoding_huffman(encoded_data1)
decompressed_text = decoding_huffman(compressed_text, tree)

print("Texte original:", encoded_data1)
print("Texte compressé:", compressed_text)
print("Texte décompressé:", decompressed_text)
print("Original == décodé: ", encoded_data1 == decompressed_text)


# In[72]:


def fusionner(strings_list):
    tailles = []
    string_fusionned = ""
    
    # Parcourir tous les éléments de la liste
    for string in strings_list:
        # Convertir chaque nombre en chaîne de caractères et l'ajouter à la chaîne fusionnée
        string_fusionned += string
        
        # Ajouter la taille de l'élément à la liste des tailles
        tailles.append(len(string))
    
    # Retourner la chaîne de caractères fusionnée et la liste des tailles
    return string_fusionned, tailles


# In[73]:


def separer(string_fusionned, tailles):
    strings_list = []
    debut = 0
    
    # Parcourir toutes les tailles dans la liste
    for taille in tailles:
        # Extraire le sous-chaîne correspondant à la taille actuelle
        sous_chaine = string_fusionned[debut:debut+taille]
        
        # Convertir la sous-chaîne en nombre et l'ajouter à la liste des nombres
        strings_list.append(sous_chaine)
        
        # Mettre à jour la position de départ pour la prochaine sous-chaîne
        debut += taille
    
    # Retourner la liste des nombres
    return strings_list


# ### (a). $Sens$ $normal$

# In[74]:


def encoding_lzw(data):
    alphabet = ['0', '1']
    data_fused, taille = fusionner(data)  # Fusionner les chaînes de caractères
    encoded_data = []

    dictionary = {}  # Initialiser le dictionnaire avec les caractères de l'alphabet spécifié
    for i, char in enumerate(alphabet):
        dictionary[char] = i

    prefix = ''
    for char in data_fused:
        new_entry = prefix + char
        if new_entry in dictionary:
            prefix = new_entry
        else:
            encoded_data.append(dictionary[prefix])
            dictionary[new_entry] = len(dictionary)
            prefix = char

    if prefix:
        encoded_data.append(dictionary[prefix])

    return encoded_data, taille


# In[75]:


def decoding_lzw(compressed_data, taille):
    alphabet = ['0', '1']
    result = []
    dictionary = {}
    current_code = len(alphabet)

    # Initialiser le dictionnaire avec les caractères de l'alphabet spécifié
    for i, char in enumerate(alphabet):
        dictionary[i] = char

    old_entry = dictionary[compressed_data[0]]
    result.append(old_entry)
    for new_entry in compressed_data[1:]:
        if new_entry in dictionary:
            entry = dictionary[new_entry]
        elif new_entry == current_code:
            entry = old_entry + old_entry[0]
        else:
            raise ValueError("Mauvaise séquence compressée")

        result.append(entry)

        # Utilisez le même dictionnaire pour la décompression
        dictionary[current_code] = old_entry + entry[0]
        current_code += 1
        old_entry = entry

    result = ''.join(result)
    result = separer(result, taille)
    
    return result


# In[76]:


binary_strings = [
    "1010", 
    "110011", 
    "111000111", 
    "10101010", 
    "111100001111", 
    "101010101010", 
    "1100110011001100", 
    "111000111000111", 
    "1010101010101010", 
    "11110000111100001111"
]


# In[77]:


# Encodage des données
encoded_data, taille = encoding_lzw(binary_strings)
print("Données encodées:", encoded_data)

# Décodage des données encodées
decoded_data = decoding_lzw(encoded_data, taille)
print("Données décodées:", decoded_data)
print("Données initiales:", binary_strings)
print("Données décodées == données initiales ?:", binary_strings == decoded_data)


# In[78]:


def intermediaire_compression(imageA, num_colorsA):
  ycrcb_imageA = rgb_to_ycrcb(imageA)
  color_paletteA, palette_indicesA = kmeans_clustering_palette(ycrcb_imageA, num_colorsA)
  mapped_indicesA = map_to_palette(ycrcb_imageA, color_paletteA)
  liste_blocsA, dimensions_originalesA = subdivision(mapped_indicesA, 16)
  compressed_textsA = []
  treesA = []
  
  for blocA in liste_blocsA:
    vecteur_comprimeA, shapeA = ligne_scan(blocA)
    chaineA = rle_encode(vecteur_comprimeA)
    compressed_textA, treeA = encoding_huffman(chaineA)
    compressed_textsA.append(compressed_textA)
    treesA.append(treeA)
    
  encoded_dataA, tailleA = encoding_lzw(compressed_textsA)
  print(encoded_dataA)
  print(encoded_dataA)
  return encoded_dataA, dimensions_originalesA, color_paletteA, palette_indicesA, shapeA, treesA, tailleA


# In[79]:


def intermediaire_decompression(encoded_dataB, dimensions_originalesB, color_paletteB, palette_indicesB, shapeB, treesB, tailleB):
  print(encoded_dataB)
  decompressed_textB = decoding_lzw(encoded_dataB, tailleB)
  liste_blocsB = []
  
  for i in range(len(decompressed_textB)):
    encoded_stringB = decoding_huffman(decompressed_textB[i], treesB[i])
    vecteur_comprimeB = rle_decode(encoded_stringB)
    blocB = inverse_ligne_scan(vecteur_comprimeB, shapeB)
    liste_blocsB.append(blocB)
  
  mapped_indicesB = reconstitution(liste_blocsB, dimensions_originalesB)
  imageB = inverse_map_to_palette(mapped_indicesB, color_paletteB)
  reconstructed_imageB = inverse_kmeans_clustering_palette(color_paletteB, palette_indicesB)
  rgb_image = ycrcb_to_rgb(imageB)
  
  return rgb_image


# In[80]:


def create_header(path, num_colors):
  image = Image.open(path)
  return intermediaire_compression(image, num_colors)[1:]


# In[81]:


def data_irm(path, num_colors):
  image = Image.open(path)
  return intermediaire_compression(image, num_colors)[0]


# In[82]:


def header_irm(path, num_colors):
    header = ""
    shapeT, color_palette, palette_indices, shapet, huffman_trT, taille = create_header(path, num_colors)
    header += f"taille: {taille}\n"
    header += f"huffman_trT: {huffman_trT}\n"
    header += f"shapeT: {shapeT}\n"
    header += f"shapet: {shapet}\n"
    header += f"color_palette: {list(map(list,list(color_palette)))}\n"
    header += f"palette_indices: {list(map(list,list(palette_indices)))}\n"
    return header


# In[83]:


def compression_irm(path1, num_colors, path2):
    header = header_irm(path1, num_colors)
    data = data_irm(path1, num_colors)
    compressed_data = "\n".join(str(nombre) for nombre in data)
    with open(path2, 'w') as f:
        f.write(header)
        f.write(compressed_data)


# In[84]:


def decompression_irm(path1, path2):
    with open(path1, 'r') as f:
        lines = f.readlines()
        numeric_lines = [line.strip() for line in lines if re.match(r'^\d+$', line.strip())]
        encoded_data = [int(nombre.strip()) for nombre in numeric_lines]
        taille = np.array(eval(lines[0].split(': ')[1]))
        huffman_trT = eval(lines[1].split(': ')[1])
        shapeT = eval(lines[2].split(': ')[1])
        shapet = eval(lines[3].split(': ')[1])
        color_palette = np.array(eval(lines[4].split(': ')[1]))
        palette_indices = np.array(eval(lines[5].split(': ')[1]))
    
    image = intermediaire_decompression(encoded_data, shapeT, color_palette, palette_indices, shapet, huffman_trT, taille)
    image.save(path2)
    return image


# In[85]:


compression_irm(path, num_colors, path1)


# In[86]:


decompression_irm(path1, path2)


# In[ ]:


img1 = Image.open(path)
img2 = Image.open(path2)


# In[ ]:


plot_images(img1, img2)


# In[ ]:


erreur = MSE(img1, img2)
print(f"L'erreur quadratique moyenne (MSE) est: {erreur}")


# In[ ]:


taux = taux_compression(path, path2)
print(f"Taux de compression: {100 * taux} %")

