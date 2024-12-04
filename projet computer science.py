import numpy as np
import trimesh
import polyscope as ps


def load_vertices(file_path):

    vertices = []
    with open(file_path, "r") as vert_file:
        for line in vert_file:
            # Ignorer les lignes de texte et vides
            if line.startswith("Sommets") or line.strip() == "":
                continue
            # Lire et convertir chaque sommet
            vertices.append([float(coord) for coord in line.split()])
    return np.array(vertices)


def load_faces(file_path):

    faces = []
    with open(file_path, "r") as face_file:
        for line in face_file:
            # Ignorer les lignes de texte et vides
            if line.startswith("Faces") or line.strip() == "":
                continue
            # Lire et convertir chaque face
            faces.append([int(index) for index in line.split()])
    return np.array(faces)


def export_mesh(vertices, faces, file_path):

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(file_path)


# Lancer le programme principal
if __name__ == "__main__":
    # Charger les sommets et faces depuis les fichiers
    vertices = load_vertices("vertices.txt")
    faces = load_faces("faces.txt")

    # Exporter le maillage au format OBJ
    export_mesh(vertices, faces, "cle.stl")


