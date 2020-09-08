# Face_recognition
Contrôle de présence automatique à l’aide des algorithmes de vision par ordinateur et d’apprentissage
Pour pallier aux appels par identifiant de façon manuelle dans beaucoup d'etablissement dont le
processus peut prendre 3 à 5 minutes. Cette perte de temps peut également engendrer
la perturbation du planning des cours, le processus d’enseignement et d’apprentissage. Mais
dans ce nouveau système, il ne faut que 1 à 2 secondes de présence.
L’objectif est de mettre en place une application capable de répondre à ces critères qu’est la
reconnaissance faciale.

Nous avons 5 scripst et le fichier encodings.pickle:
1. search_bing_api.py : l'etape 1 consiste à créer un ensemble de donnée. Pour savoir comment utiliser l'API Bing # https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/ mais j'ai déja effectué cette étape
2. encode_faces.py : (Vecteurs de 128 dimensions) Pour extraire les caractéristiques des images de visage contenues dans le dossier dataset avec leurs noms respectifs
3. recognize_faces_image.py : Reconnaitre les visages dans une seule image (en fonction de notre jeu de données)
4. recognize_faces_video.py : Reconnaitre les visages dans un flux vidéo en direct de votre webcam et produire une vidéo
5. recognize_faces_video_file.py : Reconnaitre les visages dans un fichier vidéo
6. encodings.pickle : fichier contenant les caractérisques des images contenues dans le dataset

Les 4 répertoires sont :
dataset : repertoire contenant nos images d'entrainements ainsi que leurs noms.
example : repertoire contenant nos images à tester
output  : repertoire de sortie des vidéos dans lequels les visages sont réconnues.
videos  : reportoire contenant les vidéos à entrer afin de reconnaitre les visages.

Procédures: 
1. Former votre dataset
2. Exécuter le script encode_faces.py pour extraire les caractérisques
3. le reste selon votre besoin
