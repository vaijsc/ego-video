# import cv2
# import os
# from tqdm import tqdm

# # Đường dẫn đến thư mục chứa video
# video_folder = '/home/ubuntu/video-generation/something_something_v2/data/20bn-something-something-v2'

# # Biến để đếm tổng số khung hình và số video
# total_frames = 0
# total_videos = 0
# min_frames = 99999999999
# max_frames = 0
# max_fps = 0
# min_fps = 9000000000
# # Lặp qua tất cả các file trong thư mục
# for id, filename in enumerate(tqdm(os.listdir(video_folder))):
#     if id > 50000:
#         break
#     if filename.endswith(".webm"):
#         video_path = os.path.join(video_folder, filename)
        
#         # Mở video bằng OpenCV
#         cap = cv2.VideoCapture(video_path)
        
#         # Kiểm tra xem video có mở được không
#         if not cap.isOpened():
#             print(f"Không thể mở video: {filename}")
#             continue
        
#         # Lấy số lượng khung hình trong video
#         frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
#         if min_frames > frames:
#             print(f"Min frames: {frames}, id: {filename}")
#         min_frames = min(min_frames, frames)
#         max_frames = max(max_frames, frames)
        
#         if min_fps > cv2.CAP_PROP_FPS:
#             print(f"Min FPS: {cv2.CAP_PROP_FPS}, id: {filename}")
            
#         min_fps = min(min_fps, cv2.CAP_PROP_FPS)
#         max_fps = max(max_fps, cv2.CAP_PROP_FPS)
        
#         total_frames += frames
#         total_videos += 1
        
#         # Đóng video sau khi xử lý
#         cap.release()

# # Tính số khung hình trung bình trên mỗi video
# if total_videos > 0:
#     average_frames = total_frames / total_videos
#     print(f"Trung bình số khung hình trên mỗi video: {average_frames}")
#     print(f"Ít frames nhất: {min_frames}")
#     print(f"Nhiều frames nhất: {max_frames}")
#     print(f"Min FPS: {min_fps}")
#     print(f"Max FPS: {max_fps}")

# else:
#     print("Không có video nào được tìm thấy trong thư mục.")

import decord
from decord import VideoReader
decord.bridge.set_bridge('torch')  # nếu bạn muốn làm việc với Tensor trong PyTorch

video_path = '/home/ubuntu/video-generation/something_something_v2/data/20bn-something-something-v2/1.webm'

# Khởi tạo VideoReader
try:
    vr = VideoReader(video_path, num_threads=1)
    print(f"Video có {len(vr)} khung hình")
    # Lấy khung hình đầu tiên
    # frame = vr[20]
    frame = vr.get_batch([0,1,2,3,4,5,6])
    # frame = vr.get_batch([1, 4])
    print("Frame đầu tiên:", frame.shape)
except decord.DECORDError as e:
    print(f"Không thể đọc video: {e}")
