Program służy do generowania syntetycznego dataseta.
Zrezygnowałem z biblioteki blenderproc2 ponieważ:

Główne przyczyny:
--> blenderproc2 jest świeży i ma sporo bugów
--> blenderproc2 nie obsługuje renderowania z przyspieszeniem GPU
--> blenderproc2 nie generuje "trimapów" do Salient object detection

Wersje użytych programów:
--> Blender 2.93.5
--> Python 3.9.7

Krótki opis funkcji:
1) class CustomFormatter(logging.Formatter):
po prostu kolorowy logger
2) def resize_i_relokacja()
ustawia wielkość obiektu tak, aby był mniejszy niż jeden metr. Następnie ustawia obiekt centrum XYZ (0,0,0)
3) def select_camera(kont):
wybiera kamerę o specyficznej nazwie i ustawia jej pozycje aby automatycznie kadrowała obiekt
4) def random_number():
to generator losowej pozycji dla kamery, aby pozycja kamery była losowa
5) def camera_rotation(cam_rotation) i def fixing_camera_rotation(cam_rotation)
ustawia kadr kamery
6) def save_file(path):
zapisuje projekt do scieżki
7) def change_material_type(name):
zmienia materiał obiektu plane
8) def render_to_folder_instantly(material):
natychmiast renderuje zdjęcia do folderu
9) def create_folder_and_write_blend_to_it():
tworzy nowy folder i zapisuje do niego pliki .blend, aby później można było je zrenderować
10) def renderblender(): (pliku run_blenderender.py)
Sprawdza, czy folder był już wyrenderowany.

Jak użyć tego repo?:
a) zrób konto na blenderkit.com
b) dodaj wtyczkę blender kit do blendera
c) importuj obiekty i baw się światłem i sceną, hdrami
d) zobacz zakładkę scripting w blenderze i sprawdź, czy ścieżki plików istnieją.
e) zobacz skrypt run_blenderender.py i popraw ścieżki do folderów.
f) jak chcesz od razu renderować to na końcu skryptu w zakładce scripting zakomentuj odpowiednie funkcje np. render_to_folder_instantly(material_type)
czyli:

'''
#resize_i_relokacja()

create_folder_and_write_blend_to_it()

#render_to_folder_instantly(material_type)
'''
albo jeżeli chcesz resizować matematycznie obiekt do odpowiednich wymiarów:
'''
resize_i_relokacja()

create_folder_and_write_blend_to_it()

#render_to_folder_instantly(material_type)
'''

g) użyj python3 run_blenderender.py aby wyrenderować zdjęcia.

![Alt text](50_6c1541852ee6be129ae27d4aa81.jpg?raw=true "Title")
![Alt text](50_6c1541852ee6be129ae27d4aa81.png?raw=true "Title")

