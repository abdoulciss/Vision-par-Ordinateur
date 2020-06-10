# USAGE
# python recognize_faces_video.py --encodings encodings.pickle
# python recognize_faces_video.py --encodings encodings.pickle --output output/example_output.avi --display 0

# importer les paquets nécessaires
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

# construire l'analyseur d'arguments et analyser les arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# charger les visages et les embeddings connus
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialiser le flux vidéo et le pointeur sur le fichier vidéo de sortie, puis
# laisser le capteur de l'appareil photo s'allumer
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

# boucle sur les images du flux de fichiers vidéo
while True:
	# récupérer l'image du flux vidéo fileté
	frame = vs.read()
	
	# convertir le cadre d’entrée de BGR en RGB puis redimensionner pour avoir
	# une largeur de 750px (pour accélérer le traitement)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=750)
	r = frame.shape[1] / float(rgb.shape[1])

	# détecter les coordonnées (x, y) des boîtes englobantes
	# correspondant à chaque face dans la trame en entrée, puis calcul
	# les imbrications faciales pour chaque visage
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

	# faire une boucle sur les plis du visage
	for encoding in encodings:
		# Faire correspondre chaque visage dans l'image d'entrée à notre connu
		# encodage
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"

		# vérifier si nous avons trouvé une correspondance
		if True in matches:
			# trouver les index de tous les visages correspondants puis initialiser un
			# dictionnaire pour compter le nombre total de fois chaque visage trouvé
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# boucle sur les index correspondants et maintenir un compte pour
			# chaque visage reconnu
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# déterminer la face reconnue avec le plus grand nombre
			# de vote sera selectionné la premiere entree dans le dictionnaire
			name = max(counts, key=counts.get)
		
		# Met à jour la liste de noms
		names.append(name)

	# boucle sur les visages reconnus
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# redimensionner les coordonnées du visage
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)

		# dessine le nom du visage prédit sur l'image
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

	# si le scénariste est Aucun * ET * nous sommes supposés écrire
	# la vidéo de sortie sur le disque initialise le graveur
	if writer is None and args["output"] is not None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 20,
			(frame.shape[1], frame.shape[0]), True)

	# si l'écrivain n'est pas Aucun, écrivez le cadre avec reconnaissance
	# faces sur le disque
	if writer is not None:
		writer.write(frame)

	# vérifier pour voir si nous sommes censés afficher le cadre de sortie à l'ecran
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# si la touche `q` a été enfoncée, sortir de la boucle
		if key == ord("q"):
			break

# faire un peu de nettoyage
cv2.destroyAllWindows()
vs.stop()

# vérifier si le point de l'enregistreur vidéo doit être libéré
if writer is not None:
	writer.release()