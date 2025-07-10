# 📁 File: data_split.py
import os
import glob
import shutil
import numpy as np
import logging

# Danh mục các lớp
DANH_MUC_RAC = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def tao_thu_muc_chia_du_lieu(duong_dan_goc, duong_dan_dich="dataset_split",
                             ti_le_train=0.7, ti_le_val=0.2, ti_le_test=0.1):
    """
    Chia dữ liệu từ thư mục gốc thành train/val/test và lưu vào thư mục đích
    """
    logger.info("Bắt đầu chia dữ liệu...")

    for split in ['train', 'validation', 'test']:
        for category in DANH_MUC_RAC:
            os.makedirs(os.path.join(duong_dan_dich, split, category), exist_ok=True)

    for category in DANH_MUC_RAC:
        path_category = os.path.join(duong_dan_goc, category)
        if not os.path.exists(path_category):
            logger.warning(f"Không tìm thấy thư mục: {category}")
            continue

        images = glob.glob(os.path.join(path_category, '*'))
        if len(images) == 0:
            logger.warning(f"Không có ảnh trong thư mục: {category}")
            continue

        np.random.shuffle(images)
        total = len(images)
        n_train = int(total * ti_le_train)
        n_val = int(total * ti_le_val)

        split_map = {
            'train': images[:n_train],
            'validation': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }

        for split, files in split_map.items():
            for f in files:
                shutil.copy2(f, os.path.join(duong_dan_dich, split, category, os.path.basename(f)))

        logger.info(f"{category}: Train={len(split_map['train'])}, Val={len(split_map['validation'])}, Test={len(split_map['test'])}")

    logger.info("Hoàn thành chia dữ liệu!")
    return duong_dan_dich

if __name__ == "__main__":
    # ⚠️ Nhớ chỉnh lại đường dẫn dữ liệu
    DUONG_DAN_DU_LIEU_GOC = "./dataset-resized"
    tao_thu_muc_chia_du_lieu(DUONG_DAN_DU_LIEU_GOC)  # Hoặc truyền thêm "dataset_split" nếu muốn
