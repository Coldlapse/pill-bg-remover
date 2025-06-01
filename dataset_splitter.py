import os
import shutil
import random

# 원본 폴더 경로
src_images_dir = './images'
src_masks_dir = './masks'

# 타겟 폴더 경로
dst_train_img = './train/images'
dst_train_mask = './train/masks'
dst_test_img = './test/images'
dst_test_mask = './test/masks'

# 디렉토리 생성
for d in [dst_train_img, dst_train_mask, dst_test_img, dst_test_mask]:
    os.makedirs(d, exist_ok=True)

# 이미지 파일 목록 로드 (확장자 기준 .jpg, .png 등 유연하게 대응)
image_files = [f for f in os.listdir(src_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(image_files)

# 7:3 분할
split_idx = int(len(image_files) * 0.7)
train_files = image_files[:split_idx]
test_files = image_files[split_idx:]

# 복사 함수
def move_pair(file_list, img_dst, mask_dst):
    for fname in file_list:
        src_img = os.path.join(src_images_dir, fname)
        src_mask = os.path.join(src_masks_dir, os.path.splitext(fname)[0] + '.png')  # 마스크는 .png라고 가정

        dst_img = os.path.join(img_dst, fname)
        dst_mask = os.path.join(mask_dst, os.path.basename(src_mask))

        if os.path.exists(src_mask):
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_mask, dst_mask)
        else:
            print(f"⚠️ 마스크 없음: {src_mask}")

# 실행
move_pair(train_files, dst_train_img, dst_train_mask)
move_pair(test_files, dst_test_img, dst_test_mask)

print(f"✅ 분할 완료: Train={len(train_files)}, Test={len(test_files)}")
