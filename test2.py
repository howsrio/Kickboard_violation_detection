import os
import cv2
import numpy as np
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from PIL import Image, ImageDraw, ImageFont
import shutil
import re

# Azure Custom Vision 설정
prediction_endpoint1 = "https://southcentralus.api.cognitive.microsoft.com"
prediction_key1 = "9d4f2bbf08f544b4a700983e66874ee4"
project_id_model1 = "02a5329b-ccdf-4f91-b0b3-7345baca5c4e"
model_name_model1 = "Iteration5"

prediction_endpoint2 = "https://mycustomvision-prediction.cognitiveservices.azure.com"
prediction_key2 = "55qgYtEP1itSIa15LGdJN6YWVjVCJS0rYFFr2R2sEuBMTfYvIrOjJQQJ99AJACL93NaXJ3w3AAAIACOGcmyK"
project_id_model2 = "cf9ed873-e58b-4cbf-9295-c2ba83fd0b68"
model_name_model2 = "Iteration9"

# 예측 클라이언트 생성
credentials1 = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key1})
predictor_model1 = CustomVisionPredictionClient(endpoint=prediction_endpoint1, credentials=credentials1)

credentials2 = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key2})
predictor_model2 = CustomVisionPredictionClient(endpoint=prediction_endpoint2, credentials=credentials2)

# 비디오에서 프레임 저장
def save_frames(video_path, output_dir, frame_interval):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 지정된 간격에 따라 프레임 저장
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame: {frame_filename}")

        frame_count += 1

    cap.release()

# 라벨링이 있으면 그걸 사용하고, 없으면 없는 대로 처리하는 함수
def process_frame(image_path, image, model_type):
    h, w, ch = np.array(image).shape

    # 폰트 크기 설정
    font_size = 30  # 폰트 크기 설정 (크게 만들고 싶다면 이 값을 키우세요)
    font_path = "C:/Windows/Fonts/Arial.ttf"  # 예시 경로 (시스템 환경에 맞게 수정 필요)
    font = ImageFont.truetype(font_path, font_size)  # 폰트 크기 적용

    # 모델별로 예측을 처리합니다.
    if model_type == "kick_scooter":
        # 첫 번째 모델 예측 (킥보드 감지)
        with open(image_path, mode="rb") as image_data:
            results_model1 = predictor_model1.detect_image(project_id_model1, model_name_model1, image_data)

        draw = ImageDraw.Draw(image)
        kick_scooter_detected = False  # Kick-scooter 감지 여부

        for prediction in results_model1.predictions:
            if prediction.tag_name == "Kick-scooter" and prediction.probability * 100 > 90:
                kick_scooter_detected = True
                left = prediction.bounding_box.left * w
                top = prediction.bounding_box.top * h
                width = prediction.bounding_box.width * w
                height = prediction.bounding_box.height * h
                draw.rectangle([left, top, left + width, top + height], outline='blue', width=2)
                draw.text((left, top), f'{prediction.tag_name} {prediction.probability * 100:.2f}%', fill='blue', font=font)

        return kick_scooter_detected, image  # 감지 여부와 라벨링된 이미지 반환

    elif model_type == "helmet":
        # 두 번째 모델 예측 (헬멧 감지)
        with open(image_path, mode="rb") as image_data:
            results_model2 = predictor_model2.detect_image(project_id_model2, model_name_model2, image_data)

        draw = ImageDraw.Draw(image)

        # "helmet" 태그가 90% 이상일 때
        for prediction in results_model2.predictions:
            if prediction.tag_name == "helmet" and prediction.probability * 100 > 50:
                left = prediction.bounding_box.left * image.width
                top = prediction.bounding_box.top * image.height
                width = prediction.bounding_box.width * image.width
                height = prediction.bounding_box.height * image.height
                draw.rectangle([left, top, left + width, top + height], outline='#7CFC00', width=2)  # 헬멧을 쓴 사람
                draw.text((left, top), f'{prediction.tag_name} {prediction.probability * 100:.2f}%', fill='#7CFC00', font=font)

        # "unhelmet" 태그가 90% 이상일 때 (헬멧을 쓰지 않은 사람)
        for prediction in results_model2.predictions:
            if prediction.tag_name == "unhelmet" and prediction.probability * 100 > 100:
                left = prediction.bounding_box.left * image.width
                top = prediction.bounding_box.top * image.height
                width = prediction.bounding_box.width * image.width
                height = prediction.bounding_box.height * image.height
                draw.rectangle([left, top, left + width, top + height], outline='magenta', width=2)  # 헬멧을 쓰지 않은 사람
                draw.text((left, top), f'{prediction.tag_name} {prediction.probability * 100:.2f}%', fill='magenta', font=font)

        return image  # 라벨링된 이미지 반환

# GIF 생성 함수
def create_gif_from_images(input_dir, output_gif_path, duration=100):
    image_files = sorted([os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(".jpg")], key=natural_sort_key)
    if not image_files:
        print("No images found in the specified directory.")
        return

    frames = [Image.open(image_file).convert("RGB") for image_file in image_files]
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=duration,
        loop=0
    )
    print(f"GIF saved at {output_gif_path}")

# 자연스러운 정렬을 위한 키 생성 함수
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

# 디렉토리 삭제 함수
def delete_directories(directories):
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Deleted directory: {directory}")

if __name__ == "__main__":
    video_path = "C:/roadrainger/test3.mp4"
    output_dir = "C:/roadrainger/output_frames"
    kick_scooter_dir = "C:/roadrainger/kick_scooter_frames"
    helmet_dir = "C:/roadrainger/helmet_out"
    output_gif_path = "C:/roadrainger/output_animation.gif"

    # 디렉토리 생성
    for directory in [output_dir, kick_scooter_dir, helmet_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    frame_interval = 2

    # 동영상에서 프레임 저장
    save_frames(video_path, output_dir, frame_interval)

    # 킥보드 감지 및 저장
    kick_scooter_results = []
    for image_file in os.listdir(output_dir):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(output_dir, image_file)
            detected, labeled_image = process_frame(image_path, Image.open(image_path), "kick_scooter")
            if detected:
                labeled_image.save(os.path.join(kick_scooter_dir, image_file))  # 킥보드 감지된 이미지 저장
                kick_scooter_results.append(labeled_image)

    # 헬멧 감지 및 저장
    helmet_results = []
    for image_file in os.listdir(kick_scooter_dir):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(kick_scooter_dir, image_file)
            labeled_image = process_frame(image_path, Image.open(image_path), "helmet")
            labeled_image.save(os.path.join(helmet_dir, image_file))  # 헬멧 감지된 이미지 저장
            helmet_results.append(labeled_image)

    # GIF 생성 (최종 라벨링된 이미지 사용)
    create_gif_from_images(helmet_dir, output_gif_path, duration=100)

    # 디렉토리 삭제
    # delete_directories([output_dir, kick_scooter_dir, helmet_dir])
