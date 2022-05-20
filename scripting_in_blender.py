import bpy
from mathutils import Euler
import datetime
import random
import math
import os
import secrets
import shutil
import logging
import re

counter = 0
counter2 = 0
for collection in bpy.data.collections:
   if collection.name == "trimap":
       for obj in collection.all_objects:
            rzecz_name = str(obj.name)
            counter = counter + 1
            if counter > 1:
                raise ValueError("Critical error! kilka objektów w jednym collection!")

active_material_of_plane = re.findall('"([^"]*)"', str(bpy.data.objects["Plane"].active_material))
for item in active_material_of_plane:
    counter2 = counter2 + 1
    if counter2 > 1:
        raise ValueError("Critical error! kilka nazw aktywnych materiałów w jednym objekcie!")
print(str(active_material_of_plane[0]))
material_type = str(active_material_of_plane[0])

# zmienne ktore mozna tweakowac:
# rzecz_name = str("MacBook Pro")
# material_type = str("Procedural Black Marble Tiles")
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080

# zmienne ktorych nie mozna tweakowac
saved = False
cam = bpy.data.objects["Camera"]
obj = bpy.data.objects[rzecz_name]
hash = secrets.token_hex(nbytes=16)

# logger class
class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;21m"
    green = "\x1b[1;32m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[1;34m"
    light_blue = "\x1b[1;36m"
    purple = "\x1b[1;35m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: light_blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# create logger with 'spam_application'
logger = logging.getLogger("BlenderRenderFakeDataset")
logger.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)

# wybor objektu do resiza proporcjalnego
def resize_i_relokacja():

    obj.location = (0,0,0)
    # oryginalne wymiary objektu
    current_x, current_y, current_z =  obj.dimensions
    # tworzenie listy z wymiarami obiektu
    lista = [current_x, current_y, current_z]

    # stworzenie zmiennej, znajdz najwieksza wartosc z listy
    max_value_from_list = (max(lista))

    # utworz nowa liste do appendowania danych o nowych rozmiarach obiektu.
    new_list = []

    # proporcjalny resize listy tak aby max() z listy byl rowny = 1
    if max_value_from_list > 1:
        for element in lista:
            new_list.append(element / max_value_from_list)

    if max_value_from_list < 1:
        for element in lista:
            x = element / max_value_from_list
            new_list.append(x)

    if max_value_from_list == 1:
            new_list = lista

    # nowe zmienne
    new_x, new_y, new_z = new_list

    # append nowych wyliczonych wartosci do obiektu.
    obj.dimensions = new_x, new_y, new_z

def select_camera(kont):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)

    # 0.5 to 30 stopni 1 to 60 stopni  0 to 90
    cam.rotation_mode = 'XYZ'
    cam.rotation_euler = (kont, 0, 0)

    item='MESH'

def random_number():
    return round(random.uniform(1, 6), 2)

def camera_rotation(cam_rotation):
    text = ""
    for rotation_value in [cam_rotation]:

        bpy.ops.view3d.camera_to_view_selected()

        prev_rot = cam.rotation_euler

        cam.rotation_euler = Euler((prev_rot[0], prev_rot[1], rotation_value), 'XYZ')
        text += f"{cam.location[0]:.2f}" + " " + f"{cam.location[1]:.2f}" + " " +  f"{cam.location[2]:.2f}" + " " + f"{round(cam.rotation_euler[0], 6)}" + " " + f"{round(cam.rotation_euler[1], 6)}" + " " + f"{round(cam.rotation_euler[2], 6)}" + "\n"
        print(text)

#    with open(path + "camera_positions.txt", "a") as file:
#        file.write(text)

def fixing_camera_rotation(cam_rotation):
    text = ""
    # od 0 do 6.28 to jest full obrot
    for rotation_value in [cam_rotation]:
        bpy.ops.view3d.camera_to_view_selected()
        prev_rot = cam.rotation_euler

        cam.rotation_euler = Euler((prev_rot[0], prev_rot[1], rotation_value), 'XYZ')
        text += f"{cam.location[0]:.2f}" + " " + f"{cam.location[1]:.2f}" + " " +  f"{cam.location[2]:.2f}" + " " + f"{round(cam.rotation_euler[0], 6)}" + " " + f"{round(cam.rotation_euler[1], 6)}" + " " + f"{round(cam.rotation_euler[2], 6)}" + "\n"
        print(text)
        camera_rotation(cam_rotation)


def save_file(path):
    # to zapisuje projekt .blend
    bpy.ops.wm.save_as_mainfile(filepath=path + 'blender_file.blend')

def change_material_type(name):
    mat = bpy.data.materials.get(name)
    if mat is None:
        # create material
        print("there is a problem, nie ma tego material help me")

    # Assign it to object
    if  bpy.data.objects['Plane'].data.materials:
        # assign to 1st material slot
         bpy.data.objects['Plane'].data.materials[0] = mat
    else:
        # no slots
         bpy.data.objects['Plane'].data.materials.append(mat)

def render_to_folder_instantly(material):
    for kont in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
        nazwa = str(kont) + "_" + str(hash) + "_" + str(material_type) + "_" + str(rzecz_name)+ ".png"  
        change_material_type(material)
        select_camera(round(math.radians(kont), 2))
        fixing_camera_rotation(random_number())
        # chwilowa zmiana mm lens na 40 aby obiekty ladnie sie miescily
        bpy.data.cameras['Camera'].lens = 40
        bpy.ops.render.render()
        # powrot do 50 mm default rozmiar. 
        bpy.data.cameras['Camera'].lens = 50
        shutil.move('/home/YOURUSERNAME/Pulpit/BlenderRender/image0003.png', f'/home/YOURUSERNAME/Pulpit/BlenderRender/input/{nazwa}')
        shutil.move('/home/YOURUSERNAME/Pulpit/BlenderRender/trimap0003.png', f'/home/YOURUSERNAME/Pulpit/BlenderRender/output/{nazwa}')

def create_folder_and_write_blend_to_it():
    n = 0
    while not saved:
       path = f'/home/YOURUSERNAME/Pulpit/BlenderRender/foldery_z_zapisem/{str(n)}/'

       # Check whether the specified path exists or not
       isExist = os.path.exists(path)
       n = n + 1

       if not isExist:
           logger.info(f"Folder: {str(path)} nie istnieje, robie nowy")
           os.makedirs(path)
           save_file(path)
           break

#resize_i_relokacja()

#create_folder_and_write_blend_to_it()

render_to_folder_instantly(material_type)
