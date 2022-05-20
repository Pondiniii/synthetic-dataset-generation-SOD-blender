import os
import logging
import glob


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


path_to_blender_python_script = "/home/YOURUSERNAME/Pulpit/tmp.py"
saved = False


def renderblender():
    lista_folderow = glob.glob("/home/YOURUSERNAME/Pulpit/BlenderRender/foldery_z_zapisem/*")
    for folder in lista_folderow:
        logger.info(folder)
        processed_path = folder + "/processed"
        file_to_dot_blend = folder + "/blender_file.blend"
        isExist = os.path.exists(processed_path)
        if isExist:
            logger.warning(f"Folder {processed_path} został już przerobiony")
        else:
            logger.info(f"Folder {processed_path} nie zawiera folderu processed, a więc można renderować zdjęcia")
            logger.warning("Renderuję zdjęcia...")
            os.makedirs(folder + "/processed")
            os.system(f"blender {file_to_dot_blend} --background --python {path_to_blender_python_script}")
            logger.warning(f"Zakończono renderowanie pliku: {file_to_dot_blend}")

renderblender()

