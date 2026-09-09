from PIL import Image
import pandas as pd
import os
from tqdm import tqdm



df_selected = pd.read_csv('selected_faces.csv', index_col=0)
train_df = df_selected[df_selected['split'] == 'train'].copy()
val_df = df_selected[df_selected['split'] == 'val'].copy()



# загружаем ббоксы чтобы получить координаты
bbox_full = pd.read_csv('Anno/list_bbox_celeba.txt', sep=r'\s+', header=1, index_col=0)



# Папка для обрезанных лиц
os.makedirs('train_data_cropped', exist_ok=True)
for img_name in tqdm(train_df.index):
    #берем координаты из полного ббокса
    x = bbox_full.loc[img_name, 'x_1']
    y = bbox_full.loc[img_name, 'y_1']
    w = bbox_full.loc[img_name, 'width']
    h = bbox_full.loc[img_name, 'height']

    img_path = os.path.join('train_data', img_name)
    try:
        img = Image.open(img_path)
        # вырезаем
        cropped = img.crop((x, y, x+w, y+h))
        # сохраняем в новую папку
        cropped.save(os.path.join('train_data_cropped', img_name))
    except Exception as e:
        print(f"Ошибка с {img_name}: {e}")




os.makedirs('val_data_cropped', exist_ok=True)
for img_name in tqdm(val_df.index):
    #берем координаты из полного ббокса
    x = bbox_full.loc[img_name, 'x_1']
    y = bbox_full.loc[img_name, 'y_1']
    w = bbox_full.loc[img_name, 'width']
    h = bbox_full.loc[img_name, 'height']

    img_path = os.path.join('val_data', img_name)
    try:
        img = Image.open(img_path)
        # вырезаем
        cropped = img.crop((x, y, x+w, y+h))
        # сохраняем в новую папку
        cropped.save(os.path.join('val_data_cropped', img_name))
    except Exception as e:
        print(f"Ошибка с {img_name}: {e}")