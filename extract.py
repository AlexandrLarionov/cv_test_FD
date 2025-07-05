import os
import cv2

def extract_frames(video_path, output_dir, frame_step=5):
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {video_path}")
        return
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_step == 0:
            frame_name = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Из видео {video_path} сохранено {saved_count} кадров в {output_dir}")

def process_videos(input_dir, output_parent_dir):
    if not os.path.isdir(input_dir):
        print(f"Ошибка: директория {input_dir} не существует")
        return
    
    os.makedirs(output_parent_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.mp4', '.MOV', '.mov', '.mkv', '.flv')):
            video_path = os.path.join(input_dir, filename)
            
            video_name = os.path.splitext(filename)[0]
            output_dir = os.path.join(output_parent_dir, video_name)
            
            extract_frames(video_path, output_dir)

if __name__ == "__main__":
    input_directory = "C:\\Users\\Alex\\Desktop"
    output_directory ="C:\\Users\\Alex\\Desktop\\extract"
    
    process_videos(input_directory, output_directory)
    print("Обработка завершена!")