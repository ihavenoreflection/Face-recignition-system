import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm
from collections import Counter

# 1. Загрузка данных
bbox = pd.read_csv('Anno/list_bbox_celeba.txt', sep=r'\s+', header=1, index_col=0)
attr = pd.read_csv('Anno/list_attr_celeba.txt', sep=r'\s+', header=1, index_col=0)
attr = attr.replace(-1, 0)
partition = pd.read_csv('Eval/list_eval_partition.txt', sep=r'\s+', header=0, index_col=0, names=['partition'])
identity = pd.read_csv('Anno/identity_CelebA.txt', sep=r'\s+', header=0, index_col=0, names=['identity'])

# 2. Объединяем все данные
df = bbox.join(attr).join(partition).join(identity)
print(f'Всего изображений: {len(df)}')
print(f'Всего уникальных личностей: {df["identity"].nunique()}')

# 3. Применяем критерии отбора
df = df[(df['width'] > 40) & (df['height'] > 40)]
df = df[df['Blurry'] == 0]
df = df[df['Eyeglasses'] == 0]
df = df[df['Wearing_Hat'] == 0]

print(f'После фильтрации: {len(df)} изображений')
print(f'Уникальных личностей после фильтрации: {df["identity"].nunique()}')

# 4. Разделяем по партициям
train_pool = df[df['partition'] == 0].copy()
val_pool = df[df['partition'] == 1].copy()
test_pool = df[df['partition'] == 2].copy()  # не используем

print(f'Train pool: {len(train_pool)} images, {train_pool["identity"].nunique()} identities')
print(f'Val pool: {len(val_pool)} images, {val_pool["identity"].nunique()} identities')

# 5. Сначала формируем валидационный набор (400 личностей по 5 фото)
# Считаем сколько фото у каждой личности в val_pool
val_identity_counts = val_pool.groupby('identity').size()

# Оставляем только тех, у кого есть как минимум 5 фото
val_identities_with_5 = val_identity_counts[val_identity_counts >= 5].index.tolist()
print(f'Личностей в val с >=5 фото: {len(val_identities_with_5)}')

# Если в val_pool достаточно личностей с 5+ фото, берем оттуда
if len(val_identities_with_5) >= 400:
    # Сортируем по количеству фото для детерминированности
    val_identities_sorted = val_identity_counts[val_identities_with_5].sort_values(ascending=False)
    selected_val_identities = val_identities_sorted.head(400).index.tolist()
    val_from_train = []
    print(f'Взяли {len(selected_val_identities)} личностей из val_pool')
else:
    # Если в val_pool меньше 400, добираем из train_pool
    print(f'В val_pool только {len(val_identities_with_5)} личностей с >=5 фото')
    print(f'Нужно добрать {400 - len(val_identities_with_5)} личностей из train_pool')
    
    # Берем все доступные личности из val_pool
    selected_val_identities = val_identities_with_5.copy()
    
    # Ищем в train_pool личности, которых нет в val_pool, с >=5 фото
    train_identity_counts = train_pool.groupby('identity').size()
    train_identities_with_5 = train_identity_counts[train_identity_counts >= 5].index.tolist()
    
    # Исключаем те, что уже есть в val_pool
    train_identities_for_val = [id for id in train_identities_with_5 
                               if id not in val_pool['identity'].unique()]
    
    # Сортируем по количеству фото
    train_identities_sorted = train_identity_counts[train_identities_for_val].sort_values(ascending=False)
    
    # Добираем нужное количество
    needed = 400 - len(selected_val_identities)
    additional_val_identities = train_identities_sorted.head(needed).index.tolist()
    selected_val_identities.extend(additional_val_identities)
    val_from_train = additional_val_identities
    print(f'Добрали {len(additional_val_identities)} личностей из train_pool')

print(f'Всего выбрано для val: {len(selected_val_identities)} личностей')

# Формируем val_df (по 5 фото на личность)
val_selected = []
for identity_id in selected_val_identities:
    # Берем фото из val_pool, если есть, иначе из train_pool
    if identity_id in val_pool['identity'].unique():
        group = val_pool[val_pool['identity'] == identity_id]
    else:
        group = train_pool[train_pool['identity'] == identity_id]
    
    # Берем ровно 5 фото
    sampled = group.sample(n=5, random_state=42)
    val_selected.append(sampled)

val_df = pd.concat(val_selected)
print(f'Val: {len(val_df)} images, {val_df["identity"].nunique()} unique identities')

# 6. Формируем тренировочный набор (1000 личностей по 20 фото)
# Исключаем все личности, которые уже использованы в val
train_pool = train_pool[~train_pool['identity'].isin(val_df['identity'].unique())]

# Считаем сколько фото у каждой личности в train_pool
train_identity_counts = train_pool.groupby('identity').size()

# Оставляем только тех, у кого есть как минимум 20 фото
train_identities_with_20 = train_identity_counts[train_identity_counts >= 20].index.tolist()
print(f'Личностей в train с >=20 фото: {len(train_identities_with_20)}')

if len(train_identities_with_20) < 1000:
    print(f'ОШИБКА: Недостаточно личностей для train! Нужно 1000, есть {len(train_identities_with_20)}')
    # Можно взять с меньшим количеством фото, но это нарушит условие
    # В реальности такого не должно быть, так как CelebA достаточно большой

# Сортируем по количеству фото для детерминированности
train_identities_sorted = train_identity_counts[train_identities_with_20].sort_values(ascending=False)

# Берем ровно 1000 личностей
selected_train_identities = train_identities_sorted.head(1000).index.tolist()
print(f'Выбрано для train: {len(selected_train_identities)} личностей')

# Формируем train_df (по 20 фото на личность)
train_selected = []
for identity_id in selected_train_identities:
    group = train_pool[train_pool['identity'] == identity_id]
    # Берем ровно 20 фото
    sampled = group.sample(n=20, random_state=42)
    train_selected.append(sampled)

train_df = pd.concat(train_selected)
print(f'Train: {len(train_df)} images, {train_df["identity"].nunique()} unique identities')

# 7. Проверяем пересечение личностей
train_identities_set = set(train_df['identity'].unique())
val_identities_set = set(val_df['identity'].unique())
intersection = train_identities_set & val_identities_set
print(f'Пересечение личностей между train и val: {len(intersection)}')
if len(intersection) > 0:
    print(f'ВНИМАНИЕ: Есть пересечение! {list(intersection)[:5]}...')

# 8. Копируем файлы
os.makedirs('train_data', exist_ok=True)
os.makedirs('val_data', exist_ok=True)

source_dir = 'img_celeba/'

print("Copying train data...")
for img_name in tqdm(train_df.index):
    src_path = os.path.join(source_dir, img_name)
    dst_path = os.path.join('train_data/', img_name)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)

print("Copying val data...")
for img_name in tqdm(val_df.index):
    src_path = os.path.join(source_dir, img_name)
    dst_path = os.path.join('val_data/', img_name)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)

# 9. Сохраняем информацию об отобранных данных
train_df['split'] = 'train'
val_df['split'] = 'val'
full_df = pd.concat([train_df, val_df])
full_df[['width', 'height', 'identity', 'split']].to_csv('selected_faces.csv')

print(f"Train: {len(train_df)} images, {train_df['identity'].nunique()} unique identities")
print(f"Val: {len(val_df)} images, {val_df['identity'].nunique()} unique identities")
print(f"Всего: {len(full_df)} images, {full_df['identity'].nunique()} unique identities")
print(f"Ожидалось: Train: 20000 images (1000 * 20), Val: 2000 images (400 * 5)")
print(f"Проверка: Train = {len(train_df) == 20000}, Val = {len(val_df) == 2000}")