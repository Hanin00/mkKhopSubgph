import pandas as pd
import sys
sys.path.append(r'./api/visual_genome_python_driver')
from visual_genome import api as vg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from visual_genome import api as vg
from PIL import Image as PIL_Image
import requests
from collections import Counter
from typing import List

''' 
    check obj Name in Phrase
'''
pd.set_option('display.max_rows',None)
image_id = 294
phrase = []
sumPhrase = []
regions = vg.get_region_descriptions_of_image(id=image_id)
for i in regions :
    phrase.append(i.phrase)
    sumPhrase += (i.phrase.replace(',', ' ').replace('.',' '). split(' '))
# df = pd.DataFrame(phrase)
# print(df)

pList = Counter(sumPhrase)
# freHundred = pList.most_common(100)
# print(freHundred)
cnt = 0
for i in phrase :
    if "floor" in i :
        print(i)
        cnt += 1
print(cnt)


sys.exit()