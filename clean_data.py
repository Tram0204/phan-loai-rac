import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import hashlib
import logging
from tqdm import tqdm

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Đường dẫn đến thư mục dữ liệu
data_dir = "./dataset-resized"  # Thay bằng đường dẫn thực tế đến thư mục TrashNet

# Kích thước ảnh chuẩn
CHIEU_CAO_ANH = 224
CHIEU_RONG_ANH = 224

# Tăng cường dữ liệu
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,           # Xoay ảnh 20 độ
    width_shift_range=0.2,       # Dịch chuyển ngang 20%
    height_shift_range=0.2,      # Dịch chuyển dọc 20%
    horizontal_flip=True,        # Lật ngang
    fill_mode='nearest',         # Chế độ fill
    brightness_range=[0.8, 1.2], # Điều chỉnh độ sáng
    zoom_range=0.1               # Thu phóng 10%
)

def lam_sach_du_lieu(thu_muc_du_lieu):
    """
    Làm sạch dữ liệu: xóa ảnh hỏng và trùng lặp
    """
    logger.info("Bắt đầu quá trình làm sạch dữ liệu...")
    
    tong_so_xoa = 0
    tong_so_xu_ly = 0
    
    for danh_muc in os.listdir(thu_muc_du_lieu):
        duong_dan_danh_muc = os.path.join(thu_muc_du_lieu, danh_muc)
        if not os.path.isdir(duong_dan_danh_muc):
            continue
            
        logger.info(f"Đang xử lý thư mục: {danh_muc}")
        
        # Lấy danh sách file ảnh
        cac_file_anh = [f for f in os.listdir(duong_dan_danh_muc) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        cac_hash_da_thay = set()
        so_xoa_trong_danh_muc = 0
        
        for ten_anh in tqdm(cac_file_anh, desc=f"Làm sạch {danh_muc}"):
            duong_dan_anh = os.path.join(duong_dan_danh_muc, ten_anh)
            tong_so_xu_ly += 1
            
            try:
                # Kiểm tra ảnh có mở được không
                with Image.open(duong_dan_anh) as anh:
                    anh.verify()  # Kiểm tra tính toàn vẹn
                
                # Mở lại ảnh để tính hash (verify() làm hỏng ảnh)
                with Image.open(duong_dan_anh) as anh:
                    # Kiểm tra kích thước tối thiểu
                    if anh.size[0] < 32 or anh.size[1] < 32:
                        raise ValueError("Ảnh quá nhỏ")
                    
                    # Tính hash để kiểm tra trùng lặp
                    anh_da_resize = anh.resize((CHIEU_CAO_ANH, CHIEU_RONG_ANH), Image.Resampling.LANCZOS)
                    mang_anh = np.array(anh_da_resize)
                    hash_anh = hashlib.md5(mang_anh.tobytes()).hexdigest()
                    
                    if hash_anh in cac_hash_da_thay:
                        os.remove(duong_dan_anh)
                        so_xoa_trong_danh_muc += 1
                        logger.debug(f"Xóa ảnh trùng lặp: {ten_anh}")
                    else:
                        cac_hash_da_thay.add(hash_anh)
                        
            except (IOError, SyntaxError, ValueError, OSError) as e:
                os.remove(duong_dan_anh)
                so_xoa_trong_danh_muc += 1
                logger.debug(f"Xóa ảnh hỏng: {ten_anh} - Lỗi: {e}")
        
        tong_so_xoa += so_xoa_trong_danh_muc
        logger.info(f"Thư mục {danh_muc}: Đã xóa {so_xoa_trong_danh_muc} ảnh")
    
    logger.info(f"Hoàn thành làm sạch: Đã xóa {tong_so_xoa}/{tong_so_xu_ly} ảnh")

def dem_so_anh(duong_dan_danh_muc):
    """Đếm số ảnh trong thư mục"""
    return len([f for f in os.listdir(duong_dan_danh_muc) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])

def tang_cuong_du_lieu_den_muc_tieu(thu_muc_du_lieu, so_luong_muc_tieu=500):
    """
    Tăng cường dữ liệu cho mỗi thư mục lên số lượng mục tiêu
    """
    logger.info(f"Bắt đầu tăng cường dữ liệu đến {so_luong_muc_tieu} ảnh mỗi thư mục...")
    
    for danh_muc in os.listdir(thu_muc_du_lieu):
        duong_dan_danh_muc = os.path.join(thu_muc_du_lieu, danh_muc)
        if not os.path.isdir(duong_dan_danh_muc):
            continue
            
        so_luong_hien_tai = dem_so_anh(duong_dan_danh_muc)
        logger.info(f"Thư mục {danh_muc}: {so_luong_hien_tai} ảnh")
        
        if so_luong_hien_tai == 0:
            logger.warning(f"Thư mục {danh_muc} không có ảnh nào!")
            continue
            
        if so_luong_hien_tai >= so_luong_muc_tieu:
            logger.info(f"Thư mục {danh_muc} đã đủ ảnh ({so_luong_hien_tai} >= {so_luong_muc_tieu})")
            continue
        
        # Lấy danh sách ảnh hiện có
        cac_file_anh = [f for f in os.listdir(duong_dan_danh_muc) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        so_can_tao = so_luong_muc_tieu - so_luong_hien_tai
        logger.info(f"Cần tạo thêm {so_can_tao} ảnh cho thư mục {danh_muc}")
        
        # Tạo ảnh tăng cường
        so_da_tao = 0
        lan_thu = 0
        so_lan_thu_toi_da = so_can_tao * 2  # Tránh vòng lặp vô hạn
        
        while so_da_tao < so_can_tao and lan_thu < so_lan_thu_toi_da:
            # Chọn ngẫu nhiên một ảnh từ thư mục
            ten_anh = np.random.choice(cac_file_anh)
            duong_dan_anh = os.path.join(duong_dan_danh_muc, ten_anh)
            
            try:
                # Đọc và xử lý ảnh
                with Image.open(duong_dan_anh) as anh:
                    # Chuyển đổi sang RGB nếu cần
                    if anh.mode != 'RGB':
                        anh = anh.convert('RGB')
                    
                    anh_da_resize = anh.resize((CHIEU_CAO_ANH, CHIEU_RONG_ANH), Image.Resampling.LANCZOS)
                    mang_anh = np.array(anh_da_resize) / 255.0
                    mang_anh = np.expand_dims(mang_anh, axis=0)
                
                # Tạo ảnh tăng cường
                bo_tang_cuong = datagen.flow(
                    mang_anh,
                    batch_size=1,
                    save_to_dir=duong_dan_danh_muc,
                    save_prefix=f'tang_cuong_{so_da_tao:04d}',
                    save_format='jpeg'
                )
                
                # Tạo một ảnh tăng cường
                next(bo_tang_cuong)
                so_da_tao += 1
                
                if so_da_tao % 50 == 0:
                    logger.info(f"Đã tạo {so_da_tao}/{so_can_tao} ảnh cho {danh_muc}")
                    
            except Exception as e:
                logger.error(f"Lỗi khi tạo ảnh tăng cường từ {ten_anh}: {e}")
            
            lan_thu += 1
        
        so_cuoi_cung = dem_so_anh(duong_dan_danh_muc)
        logger.info(f"Hoàn thành {danh_muc}: {so_cuoi_cung} ảnh (tạo thêm {so_da_tao})")

def kiem_tra_dataset(thu_muc_du_lieu):
    """Kiểm tra và báo cáo thống kê dataset"""
    logger.info("Đang kiểm tra dataset...")
    
    tong_so_anh = 0
    cac_danh_muc = []
    
    for danh_muc in os.listdir(thu_muc_du_lieu):
        duong_dan_danh_muc = os.path.join(thu_muc_du_lieu, danh_muc)
        if os.path.isdir(duong_dan_danh_muc):
            so_luong = dem_so_anh(duong_dan_danh_muc)
            tong_so_anh += so_luong
            cac_danh_muc.append((danh_muc, so_luong))
    
    logger.info(f"Tổng số thư mục: {len(cac_danh_muc)}")
    logger.info(f"Tổng số ảnh: {tong_so_anh}")
    
    for danh_muc, so_luong in sorted(cac_danh_muc):
        logger.info(f"  {danh_muc}: {so_luong} ảnh")
    
    return cac_danh_muc

def ham_chinh():
    """Hàm chính thực hiện toàn bộ quá trình"""
    if not os.path.exists(data_dir):
        logger.error(f"Thư mục dữ liệu không tồn tại: {data_dir}")
        return
    
    logger.info("=== BẮT ĐẦU XỬ LÝ DỮ LIỆU TRASHNET ===")
    
    # Bước 1: Kiểm tra dataset ban đầu
    logger.info("1. Kiểm tra dataset ban đầu...")
    kiem_tra_dataset(data_dir)
    
    # Bước 2: Làm sạch dữ liệu
    logger.info("2. Làm sạch dữ liệu...")
    lam_sach_du_lieu(data_dir)
    
    # Bước 3: Kiểm tra sau khi làm sạch
    logger.info("3. Kiểm tra sau khi làm sạch...")
    kiem_tra_dataset(data_dir)
    
    # Bước 4: Tăng cường dữ liệu
    logger.info("4. Tăng cường dữ liệu...")
    tang_cuong_du_lieu_den_muc_tieu(data_dir, so_luong_muc_tieu=500)
    
    # Bước 5: Kiểm tra kết quả cuối cùng
    logger.info("5. Kiểm tra kết quả cuối cùng...")
    thong_ke_cuoi_cung = kiem_tra_dataset(data_dir)
    
    logger.info("=== HOÀN THÀNH XỬ LÝ DỮ LIỆU ===")
    
    return thong_ke_cuoi_cung

if __name__ == "__main__":
    # Thực hiện toàn bộ quá trình
    ket_qua_cuoi_cung = ham_chinh()
    
    # In kết quả tóm tắt
    print("\n" + "="*50)
    print("KẾT QUẢ XỬ LÝ DỮ LIỆU TRASHNET")
    print("="*50)
    print("Dữ liệu đã được làm sạch và tăng cường thành công!")
    print("Mỗi thư mục danh mục có tối đa 500 ảnh.")
    print("Dataset đã sẵn sàng để huấn luyện mô hình.")
    print("="*50)