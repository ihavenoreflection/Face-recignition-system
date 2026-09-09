import pandas as pd
import os
import numpy as np

def split_and_recalculate_landmarks(original_landmarks_file, 
                                    bbox_file,
                                    train_img_dir, val_img_dir, 
                                    output_dir):
    """
    Разделяет аннотации на train и val и пересчитывает координаты
    для обрезанных по bounding box изображений
    """
    
    # Загружаем оригинальные аннотации (координаты на сырых изображениях)
    annotations = pd.read_csv(
        original_landmarks_file,
        sep=r'\s+',
        skiprows=2,
        header=None,
        names=['filename', 'lefteye_x', 'lefteye_y', 'righteye_x', 'righteye_y',
               'nose_x', 'nose_y', 'leftmouth_x', 'leftmouth_y', 
               'rightmouth_x', 'rightmouth_y']
    )
    
    # Загружаем bounding box
    bbox = pd.read_csv(
        bbox_file,
        sep=r'\s+',
        skiprows=2,
        header=None,
        names=['filename', 'x_1', 'y_1', 'width', 'height']
    )
    
    train_files = set(os.listdir(train_img_dir))
    val_files = set(os.listdir(val_img_dir))
    
    def recalculate_coords_for_set(file_set, set_name):
        set_annotations = annotations[annotations['filename'].isin(file_set)].copy()
        
        if len(set_annotations) == 0:
            print(f"нет файлов для {set_name}")
            return set_annotations
        
        for idx, row in set_annotations.iterrows():
            filename = row['filename']
            bbox_row = bbox[bbox['filename'] == filename]
            if len(bbox_row) == 0:
                print(f"нет bbox для {filename}")
                continue
                
            bbox_row = bbox_row.iloc[0]
            x1 = bbox_row['x_1']
            y1 = bbox_row['y_1']

            point_names = ['lefteye_x', 'lefteye_y', 'righteye_x', 'righteye_y',
                          'nose_x', 'nose_y', 'leftmouth_x', 'leftmouth_y', 
                          'rightmouth_x', 'rightmouth_y']
            
            for i, point_name in enumerate(point_names):
                if i % 2 == 0:  # x координата
                    set_annotations.at[idx, point_name] = row[point_name] - x1
                else:  # y координата
                    set_annotations.at[idx, point_name] = row[point_name] - y1
        
        return set_annotations
    
    # Разделяем и пересчитываем координаты
    train_annotations = recalculate_coords_for_set(train_files, "train")
    val_annotations = recalculate_coords_for_set(val_files, "val")
    
    # Сохраняем результаты
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохраняем train аннотации
    train_annotations.to_csv(
        os.path.join(output_dir, 'train_landmarks.txt'),
        sep=' ',
        index=False
    )
    
    # Сохраняем val аннотации
    val_annotations.to_csv(
        os.path.join(output_dir, 'val_landmarks.txt'),
        sep=' ',
        index=False
    )
    return train_annotations, val_annotations

train_anno, val_anno = split_and_recalculate_landmarks(
    original_landmarks_file="Anno/list_landmarks_celeba.txt",
    bbox_file="Anno/list_bbox_celeba.txt",
    train_img_dir="train_data_cropped",
    val_img_dir="val_data_cropped",
    output_dir="split_landmarks_cropped"
)