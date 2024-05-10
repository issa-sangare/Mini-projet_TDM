from IPython.display import display
from PIL import Image
import numpy as np

def ext_chr(chaine):
  occ = {str(c): chaine.count(c) for c in set(chaine)}
  return sorted(occ.items(), key = lambda x: x[1], reverse = True)

def dictionnaire_huffman(noeud, binstring = ''):
  if isinstance(noeud, str):
    return {noeud: binstring}
  (l, r) = noeud
  d = {}
  d.update(dictionnaire_huffman(l, binstring + '0'))
  d.update(dictionnaire_huffman(r, binstring + '1'))
  return d

def arbre_huffman(chaine):
  noeuds = ext_chr(chaine)
  
  while len(noeuds) > 1:
    (key1, c1) = noeuds.pop()
    (key2, c2) = noeuds.pop()
    noeud = (key1, key2)
    noeuds.append((noeud, c1 + c2))
    noeuds.sort(key = lambda x: x[1], reverse = True)
    
  return noeuds[0][0]

def compress_huffman(chaine):
  noeuds = arbre_huffman(chaine)
  codeHuffman = dictionnaire_huffman(noeuds)
  chaine_compresse = ''
  
  for char in chaine:
    chaine_compresse += codeHuffman[char]
  
  return chaine_compresse

def vecteur_ligne(matrice):
  vecteur = np.array(matrice, dtype=np.uint8)
  vecteur = vecteur.flatten()
  return vecteur.tolist()

def inverse_vecteur_ligne(vecteur, lignes, colonnes):
    expected_size = lignes * colonnes
    actual_size = len(vecteur)
    
    if expected_size != actual_size:
        raise ValueError(f"Cannot reshape array of size {actual_size} into shape ({lignes},{colonnes})")
    
    mat = np.array(vecteur).reshape((lignes, colonnes))
    #return np.array(mat, dtype=np.uint8)
    return mat

def vecteur_colonne(matrice):
    vecteur = np.ravel(matrice, order = 'F')
    vecteur = vecteur.tolist()
    return vecteur

def inverse_vecteur_colonne(vecteur, lignes, colonnes):
    expected_size = lignes * colonnes
    actual_size = len(vecteur)
    
    if expected_size!= actual_size:
        raise ValueError(f"Cannot reshape array of size {actual_size} into shape ({lignes},{colonnes})")
    
    mat = np.array(vecteur).reshape((lignes, colonnes))
    #return np.array(mat, dtype=np.uint8)
    return mat

def vecteur_zigzag(matrice):
    matrice = np.array(matrice)
    return np.concatenate([np.diagonal(matrice[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1 - matrice.shape[0], matrice.shape[0])])

def inverse_vecteur_zigzag(vector, rows, cols):
    mat = [[None] * cols for _ in range(rows)]
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

    #return np.array(mat, dtype=np.uint8)
    return mat

def matrices_vecteur_ligne(matrices):
  # Assuming matrices is of shape (height, width, 3)
    R = matrices[:, :, 0]  # Extract Red channel
    G = matrices[:, :, 1]  # Extract Green channel
    B = matrices[:, :, 2]  # Extract Blue channel
    
    rouge = vecteur_ligne(R)
    vert = vecteur_ligne(G)
    bleu = vecteur_ligne(B)
    
    # Combine the vectors from each channel
    rgb = np.concatenate((rouge, vert, bleu))
    
    return rgb.tolist()

def inverse_vecteur_ligne_matrices(decompressed, L, C):
    if len(decompressed) % 3 != 0:
        raise ValueError("Decompressed data length is not a multiple of 3")

    # Division de la donnée décompressée en trois canaux de couleur
    R = decompressed[:len(decompressed)//3]
    G = decompressed[len(decompressed)//3:2*len(decompressed)//3]
    B = decompressed[2*len(decompressed)//3:]
    
    # Inversion de chaque canal de couleur
    R_inverse = inverse_vecteur_ligne(R, L, C)
    G_inverse = inverse_vecteur_ligne(G, L, C)
    B_inverse = inverse_vecteur_ligne(B, L, C)

    # Création de la matrice tridimensionnelle
    #matrice_tridim = np.array([R_inverse, G_inverse, B_inverse], dtype=np.uint8)
    matrice_tridim = np.stack((R_inverse, G_inverse, B_inverse), axis=-1)
    return np.array(matrice_tridim, dtype=np.uint8)

def matrices_vecteur_colonne(matrices):
  # Assuming matrices is of shape (height, width, 3)
    R = matrices[:, :, 0]  # Extract Red channel
    G = matrices[:, :, 1]  # Extract Green channel
    B = matrices[:, :, 2]  # Extract Blue channel
    
    rouge = vecteur_colonne(R)
    vert = vecteur_colonne(G)
    bleu = vecteur_colonne(B)
    
    # Combine the vectors from each channel
    rgb = np.concatenate((rouge, vert, bleu))
    
    return rgb.tolist()

def inverse_vecteur_colonne_matrices(decompressed, L, C):
    if len(decompressed) % 3 != 0:
        raise ValueError("Decompressed data length is not a multiple of 3")

    # Division de la donnée décompressée en trois canaux de couleur
    R = decompressed[:len(decompressed)//3]
    G = decompressed[len(decompressed)//3:2*len(decompressed)//3]
    B = decompressed[2*len(decompressed)//3:]

    # Inversion de chaque canal de couleur
    R_inverse = inverse_vecteur_colonne(R, L, C)
    G_inverse = inverse_vecteur_colonne(G, L, C)
    B_inverse = inverse_vecteur_colonne(B, L, C)

    # Création de la matrice tridimensionnelle
    #matrice_tridim = np.array([R_inverse, G_inverse, B_inverse], dtype=np.uint8)
    matrice_tridim = np.stack((R_inverse, G_inverse, B_inverse), axis=-1)
    return np.array(matrice_tridim, dtype=np.uint8)

def matrices_vecteur_zigzag(matrices):
    # Assuming matrices is of shape (height, width, 3)
    R = matrices[:, :, 0]  # Extract Red channel
    G = matrices[:, :, 1]  # Extract Green channel
    B = matrices[:, :, 2]  # Extract Blue channel
    
    rouge = vecteur_zigzag(R)
    vert = vecteur_zigzag(G)
    bleu = vecteur_zigzag(B)
    
    # Combine the vectors from each channel
    rgb = np.concatenate((rouge, vert, bleu))
    
    return rgb.tolist()

def inverse_vecteur_zigzag_matrices(decompressed, L, C):
    if len(decompressed) % 3 != 0:
        raise ValueError("Decompressed data length is not a multiple of 3")

    # Division de la donnée décompressée en trois canaux de couleur
    R = decompressed[:len(decompressed)//3]
    G = decompressed[len(decompressed)//3:2*len(decompressed)//3]
    B = decompressed[2*len(decompressed)//3:]

    # Inversion de chaque canal de couleur
    R_inverse = inverse_vecteur_zigzag(R, L, C)
    G_inverse = inverse_vecteur_zigzag(G, L, C)
    B_inverse = inverse_vecteur_zigzag(B, L, C)

    # Création de la matrice tridimensionnelle
    #matrice_tridim = np.array([R_inverse, G_inverse, B_inverse], dtype=np.uint8)
    matrice_tridim = np.stack((R_inverse, G_inverse, B_inverse), axis=-1)
    return np.array(matrice_tridim, dtype=np.uint8)

def matrices_vecteur_ligne_colonne_zigzag(matrices):
  # Assuming matrices is of shape (height, width, 3)
    R = matrices[:, :, 0]  # Extract Red channel
    G = matrices[:, :, 1]  # Extract Green channel
    B = matrices[:, :, 2]  # Extract Blue channel
    
    rouge = vecteur_ligne(R)
    vert = vecteur_colonne(G)
    bleu = vecteur_zigzag(B)
    
    # Combine the vectors from each channel
    rgb = np.concatenate((rouge, vert, bleu))
    
    return rgb.tolist()

def inverse_vecteur_ligne_colonne_zigzag_matrices(decompressed, L, C):
    if len(decompressed) % 3 != 0:
        raise ValueError("Decompressed data length is not a multiple of 3")

    # Division de la donnée décompressée en trois canaux de couleur
    R = decompressed[:len(decompressed)//3]
    G = decompressed[len(decompressed)//3:2*len(decompressed)//3]
    B = decompressed[2*len(decompressed)//3:]

    # Inversion de chaque canal de couleur
    R_inverse = inverse_vecteur_ligne(R, L, C)
    G_inverse = inverse_vecteur_colonne(G, L, C)
    B_inverse = inverse_vecteur_zigzag(B, L, C)

    # Création de la matrice tridimensionnelle
    #matrice_tridim = np.array([R_inverse, G_inverse, B_inverse], dtype=np.uint8)
    matrice_tridim = np.stack((R_inverse, G_inverse, B_inverse), axis=-1)
    return np.array(matrice_tridim, dtype=np.uint8)

def matrices_vecteur_zigzag_colonne_ligne(matrices):
  # Assuming matrices is of shape (height, width, 3)
    R = matrices[:, :, 0]  # Extract Red channel
    G = matrices[:, :, 1]  # Extract Green channel
    B = matrices[:, :, 2]  # Extract Blue channel
    
    rouge = vecteur_zigzag(R)
    vert = vecteur_colonne(G)
    bleu = vecteur_ligne(B)
    
    # Combine the vectors from each channel
    rgb = np.concatenate((rouge, vert, bleu))
    
    return rgb.tolist()

def inverse_vecteur_zigzag_colonne_ligne_matrices(decompressed, L, C):
    if len(decompressed) % 3 != 0:
        raise ValueError("Decompressed data length is not a multiple of 3")

    # Division de la donnée décompressée en trois canaux de couleur
    R = decompressed[:len(decompressed)//3]
    G = decompressed[len(decompressed)//3:2*len(decompressed)//3]
    B = decompressed[2*len(decompressed)//3:]

    # Inversion de chaque canal de couleur
    R_inverse = inverse_vecteur_zigzag(R, L, C)
    G_inverse = inverse_vecteur_colonne(G, L, C)
    B_inverse = inverse_vecteur_ligne(B, L, C)

    # Création de la matrice tridimensionnelle
    #matrice_tridim = np.array([R_inverse, G_inverse, B_inverse], dtype=np.uint8)
    matrice_tridim = np.stack((R_inverse, G_inverse, B_inverse), axis=-1)
    return np.array(matrice_tridim, dtype=np.uint8)

from PIL import Image

def RGB_grayscale_vector(image_path, file_path):
    # Ouvrir l'image avec Pillow
    img = Image.open(image_path)
    print(img.size)

    # Obtenir le mode de l'image
    mode = img.mode
    img = np.array(img, dtype=np.uint8)
    
    if file_path == 'l':
      if mode == 'RGB':
        return matrices_vecteur_ligne(img)
      elif mode == 'L':
        return vecteur_ligne(img)
      else:
        print("Mode de fichier non reconnu")
        return None
    elif file_path == 'c':
      if mode == 'RGB':
        return matrices_vecteur_colonne(img)
      elif mode == 'L':
        return vecteur_colonne(img)
      else:
        print("Mode de fichier non reconnu")
        return None
    elif file_path == 'z':
      if mode == 'RGB':
        return matrices_vecteur_zigzag(img)
      elif mode == 'L':
        return vecteur_zigzag(img)
      else:
        print("Mode de fichier non reconnu")
        return None
    elif file_path == 'lcz':
      if mode == 'RGB':
        return matrices_vecteur_ligne_colonne_zigzag(img)
      else:
        print("L'image n'est pas en mode RGB: elle ne peut pas être parcourue en lcz")
        return None
    elif file_path == 'zcl':
      if mode == 'RGB':
        return matrices_vecteur_zigzag_colonne_ligne(img)
      else:
        print("L'image n'est pas en mode RGB: elle ne peut pas être parcourue en zcl")
        return None
    else:
      print("Type de parcours inconnue")
      
def RGB_grayscale_binary(path, file_path):
  img = RGB_grayscale_vector(path, file_path)
  if img == None:
    return None
  img = ''.join(format(byte, '08b') for byte in img)
  return img

def binary_matrix_to_vector(path, file_path):
    # Ouvrir l'image avec Pillow
    img = Image.open(path)

    # Obtenir le mode de l'image
    mode = img.mode
    img = np.array(img, dtype=np.uint8)
    
    if mode != '1':
      print("Le mode du fichier n'est pas binaire")
      return None
    
    if file_path == 'l':
        return vecteur_ligne(img)
    elif file_path == 'c':
        return vecteur_colonne(img)
    elif file_path == 'z':
        return vecteur_zigzag(img)
    else:
      print("Type de parcours inconnue")
      return None, None, None
  
def binary_binary(path, file_path):
    img = binary_matrix_to_vector(path, file_path)
    if img == None:
        return None
    img = ''.join(str(byte) for byte in img)
    return img

def binary_to(binary_sequence, mode, dimensions, traversal_type):
    img_data = []
    # Convertir la séquence binaire en une liste d'entiers représentant chaque bloc de 8 bits
    integer_list = [int(binary_sequence[i:i+8], 2) for i in range(0, len(binary_sequence), 8)]

    # Vérification du mode de fichier
    if mode == '1':
        integer_list = [int(i) for i in binary_sequence]
        if traversal_type == 'l':
            img_data = inverse_vecteur_ligne(integer_list, dimensions[1], dimensions[0])
        if traversal_type == 'c':
            img_data = inverse_vecteur_colonne(integer_list, dimensions[1], dimensions[0])
        if traversal_type == 'z':
            img_data = inverse_vecteur_zigzag(integer_list, dimensions[1], dimensions[0])
    elif mode == 'L':
        if traversal_type == 'l':
            img_data = inverse_vecteur_ligne(integer_list, dimensions[1], dimensions[0])
        if traversal_type == 'c':
            img_data = inverse_vecteur_colonne(integer_list, dimensions[1], dimensions[0])
        if traversal_type == 'z':
            img_data = inverse_vecteur_zigzag(integer_list, dimensions[1], dimensions[0])
    elif mode == 'RGB':
        if traversal_type == 'l':
            img_data = inverse_vecteur_ligne_matrices(integer_list, dimensions[1], dimensions[0])
        elif traversal_type == 'c':
            img_data = inverse_vecteur_colonne_matrices(integer_list, dimensions[1], dimensions[0])
        elif traversal_type == 'z':
            img_data = inverse_vecteur_zigzag_matrices(integer_list, dimensions[1], dimensions[0])
        elif traversal_type == 'lcz':
            img_data = inverse_vecteur_ligne_colonne_zigzag_matrices(integer_list, dimensions[1], dimensions[0])
        elif traversal_type == 'zcl':
            img_data = inverse_vecteur_zigzag_colonne_ligne_matrices(integer_list, dimensions[1], dimensions[0])
        else:
            print("Type de parcours inconnu")
            return
    else:
        print("Mode de fichier inconnu")
        return
    
    return img_data