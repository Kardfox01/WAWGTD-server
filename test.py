
from PIL import Image

from neuro import Neuro
from models import *
import prms

neuro = Neuro()
def test():

    img = "image.png"
    img_user, img_ollama = neuro.depth_marked(Image.open(img))

    if not neuro.tont(img_ollama): print("fuck u")
    else: print("derevo")

if __name__ == "__main__":
    test()