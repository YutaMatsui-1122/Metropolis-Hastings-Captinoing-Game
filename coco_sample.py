from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import random
from PIL import Image

# COCOデータセットの初期化
dataDir = 'dataset/coco'
dataType = 'train2014'
annFile = f'{dataDir}/annotations/annotations/captions_{dataType}.json'

# COCO APIの初期化
coco = COCO(annFile)

# 画像IDの取得
imgIds = coco.getImgIds()
selectedImgId = random.choice(imgIds)
print("selected image id: ", selectedImgId)
img = coco.loadImgs(selectedImgId)[0]

print("image info: ", img)

# 画像の読み込みと表示
img_path = f'{dataDir}/{dataType}/{img["file_name"]}'
image = Image.open(img_path)
plt.imshow(image)
plt.axis('off')

# 関連するキャプションの取得と表示
print("image id: ", img['id'])
annIds = coco.getAnnIds(imgIds=img['id'])
print("annotation ids: ", annIds)
anns = coco.loadAnns(annIds)
print("annotations: ", anns)
for ann in anns:
    print(ann['caption'])

plt.savefig('coco_sample.png')