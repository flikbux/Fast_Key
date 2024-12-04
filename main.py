import cv2
import numpy as np
from PIL import Image as im
from matplotlib import pyplot as plt


def key_detector(image_path):

    def initialize(img_path):
        # Read the original image
        img = cv2.imread(img_path)
        # Display original image
        # Display the image using Matplotlib
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Conversion BGR -> RGB pour Matplotlib
        plt.title("Image originale")
        plt.axis("off")
        plt.show()
        return img


    def detect_border(img):
        # Convert to graycsale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Blur the image for better edge detection
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

        # Canny Edge Detection
        edges = cv2.Canny(image=img_blur, threshold1=10, threshold2=250)  # Canny Edge Detection
        # Display Canny Edge Detection Image
        cv2.waitKey(0)
        edgepic = im.fromarray(edges)
        edgepic.save('edgepic.png')
        cv2.destroyAllWindows()
        return edgepic

        #----------------------- partie detect circle
    def detect_circle(edgepic):
        edges = np.array(edgepic)
        if edges is not None:
            # Detect circles using Hough Transform
            detected_circles = cv2.HoughCircles(
                edges,
                cv2.HOUGH_GRADIENT,  # Corrected method
                dp=1,  # Inverse ratio of the accumulator resolution to the image resolution
                minDist=10,  # Minimum distance between detected centers
                param1=70,  # Upper threshold for the Canny edge detector
                param2=30,  # Accumulator threshold for the circle centers at the detection stage
                minRadius=10,  # Minimum circle radius
                maxRadius=103   # Maximum circle radius
            )

            # If some circles are detected, draw the biggest
            if detected_circles is not None:
                detected_circles = np.uint16(np.around(detected_circles))  # Round to integers
                amax, bmax, rmax = 0, 0, 0
                for pt in detected_circles[0, :]:  # Loop through the detected circles
                    a, b, r = pt[0], pt[1], pt[2]  # Extract circle parameters
                    if rmax < r:
                        amax, bmax, rmax = a, b, r  # Extract circle parameters
                # Display the result
                # Draw the circumference of the circle.
                cv2.circle(img, (amax, bmax), rmax, (0, 255, 0), 2)

                # Draw a small circle (of radius 1) to show the center.
                # Draw a circle on the image
                cv2.circle(img, (amax, bmax), 1, (0, 0, 255), 3)

                # Display the image with the circle using Matplotlib
                plt.figure(figsize=(8, 8))
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Conversion BGR -> RGB pour Matplotlib
                plt.title("Detected Circle")
                plt.axis("off")
                plt.show()
                return amax, bmax, rmax
            else:
                return 0
        else:
            return 0
        #----------------------- partie scaling
    def scale_pixel_mm(r):
        # DEFINITION DU RATION PIXEL ---> mm
        # RAYON PIECE 1mm : 11,625mm
        # dimension mm en pixel : 1 mm = npixel*rayonPiece/rayon
        rayonPiece1euro = 11.625
        scale_mm_to_pix = rayonPiece1euro / r
        return scale_mm_to_pix
        #----------------------- partie detect region
    def detect_color(img, a, b, r):
        #Calcule la couleur moyenne des pixels dans un cercle et retourne une liste des positions des pixels.

        # Assurer que l'image est un numpy array
        img = np.array(img)

        # Obtenir les dimensions de l'image
        h, w = img.shape[:2]

        # Liste pour stocker les positions des pixels
        liste_positions = []

        # Masque pour définir le cercle
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (a, b), r, 255, thickness=-1)  # Dessiner un cercle blanc plein

        # Parcourir les pixels dans l'image
        for y in range(max(0, b - r), min(h, b + r + 1)):
            for x in range(max(0, a - r), min(w, a + r + 1)):
                if mask[y, x] == 255:  # Si le pixel est dans le cercle
                    liste_positions.append((x, y))

        # Extraire les pixels de l'image selon le masque
        pixels = img[mask == 255]

        # Calculer la couleur moyenne
        couleur_moyenne = tuple(np.mean(pixels, axis=0).astype(int))
        return couleur_moyenne, liste_positions
        #----------------------- partie detect clé


    def detect_clé(img, couleur_moyenne, liste_positions, treshold):
        # Assurer que l'image est un numpy array
        img = np.array(img)
        img_visual = img.copy()  # Copie pour visualisation

        # Convertir la couleur moyenne en un tableau numpy
        couleur_moyenne_np = np.array(couleur_moyenne, dtype=np.float32)

        # Créer un masque pour les pixels déjà présents dans liste_positions
        masque_existant = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for x, y in liste_positions:
            masque_existant[y, x] = 1

        # Calculer la distance euclidienne entre chaque pixel et la couleur moyenne
        distances = np.linalg.norm(img.astype(np.float32) - couleur_moyenne_np, axis=2)

        # Créer un masque pour les pixels proches de la couleur moyenne
        masque_couleur = distances <= treshold

        # Exclure les pixels déjà dans liste_positions en combinant les masques
        masque_final = masque_couleur & ~masque_existant.astype(bool)

        # Récupérer les coordonnées des pixels détectés
        positions_bout_clé = np.column_stack(np.where(masque_final))

        # Colorer les pixels détectés en rouge pour visualisation
        img_visual[masque_final] = [0, 0, 255]

        # Afficher l'image avec les pixels détectés en rouge
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(img_visual, cv2.COLOR_BGR2RGB))  # Conversion BGR -> RGB pour Matplotlib
        plt.title("Pixels détectés colorisés en rouge")
        plt.axis("off")
        plt.show()

        return [tuple(pos) for pos in positions_bout_clé]


        # Afficher l'image avec les pixels détectés en rouge
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(img_visual, cv2.COLOR_BGR2RGB))  # Conversion BGR -> RGB pour Matplotlib
        plt.title("Pixels détectés colorisés en rouge")
        plt.axis("off")
        plt.show()

        return positions_bout_clé



    # Call edge detection function
    img = initialize(img_path)
    edges = detect_border(img)
    a, b, r = detect_circle(edges)
    scale = scale_pixel_mm(r)
    print("ratio mm -> pixel :", scale,"mm par pixel")
    couleur_moyenne, liste_positions = detect_color(img, a, b, r)
    print( "la couleur moyenne de la clé est :", couleur_moyenne)
    liste_positions = detect_clé(img, couleur_moyenne, liste_positions, 100)
    print(len(liste_positions))
    return liste_positions

def mise_en_matrice(liste_positions, precision, layer):
    def suppression_points(liste_positions, precision):
        liste_positions_filtrée = []
        for x, y in liste_positions:
            # Ajouter uniquement si les coordonnées sont des multiples de `i`
            if x % precision == 0 and y % precision == 0:
                liste_positions_filtrée.append((x, y))
        return liste_positions_filtrée
    def mise_en_3D(liste_positions, precision, layer):
        num_layers = layer
        layer_height = precision
        matrice_points = []
        for i in range(len(liste_positions)):
            x = liste_positions[i][0]
            y = liste_positions[i][1]
            for z in np.arange(0, num_layers * layer_height, layer_height):  # Utilisation de numpy.arange
                matrice_points.append((x, y, z))
        print(matrice_points[:100])
        print(len(matrice_points))
        return matrice_points
    def mise_en_txt(matrice_points, precision, layer):
        txt_output_path = f'coordonnees_cle_precision{precision}_layer{layer}.txt'
        with open(txt_output_path, 'w') as text_file:
            # Écrire les coordonnées sans parenthèses
            for i in range(len(matrice_points)):
                # Formatage des coordonnées en chaîne sans parenthèses
                x, y, z = matrice_points[i]
                text_file.write(f"{x}, {y}, {z}\n")
    liste_positions = suppression_points(liste_positions, precision)
    matrice_points = mise_en_3D(liste_positions, precision, layer)
    mise_en_txt(matrice_points, precision, layer)

img_path = 'test.jpg'
liste_positions = key_detector(img_path)
mise_en_matrice(liste_positions, 4, 2)