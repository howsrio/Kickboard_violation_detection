import os
import cv2
import asyncio
import aiohttp
import shutil
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from PIL import Image, ImageDraw
import re

# Azure Custom Vision 설정 - 킥보드
prediction_endpoint1 = "https://southcentralus.api.cognitive.microsoft.com"
prediction_key1 = "9d4f2bbf08f544b4a700983e66874ee4"
project_id_model1 = "02a5329b-ccdf-4f91-b0b3-7345baca5c4e"
model_name_model1 = "Iteration5"

# Azure Custom Vision 설정 - 헬멧
prediction_endpoint2 = "https://mycustomvision-prediction.cognitiveservices.azure.com"
prediction_key2 = "55qgYtEP1itSIa15LGdJN6YWVjVCJS0rYFFr2R2sEuBMTfYvIrOjJQQJ99AJACL93NaXJ3w3AAAIACOGcmyK"
project_id_model2 = "cf9ed873-e58b-4cbf-9295-c2ba83fd0b68"
model_name_model2 = "Iteration7"

# 파일명에 포함된 숫자를 기준으로 자연스러운 정렬을 위한 함수
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

# 예측 클라이언트 생성
credentials1 = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key1})
predictor_model1 = CustomVisionPredictionClient(endpoint=prediction_endpoint1, credentials=credentials1)

credentials2 = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key2})
predictor_model2 = CustomVisionPredictionClient(endpoint=prediction_endpoint2, credentials=credentials2)

# 첫 번째 모델에서 킥보드 예측
async def async_detect_objects(image_path, predictor_endpoint, api_key, project_id, model_name, tag_name, output_dir, color, probability_threshold):
    async with aiohttp.ClientSession() as session:
        image = Image.open(image_path)
        w, h = image.size
        draw = ImageDraw.Draw(image)

        with open(image_path, "rb") as image_data:
            payload = image_data.read()
            headers = {
                "Prediction-key": api_key,
                "Content-Type": "application/octet-stream"
            }
            async with session.post(
                f"{predictor_endpoint}/customvision/v3.0/Prediction/{project_id}/detect/iterations/{model_name}/image",
                data=payload,
                headers=headers
            ) as response:
                response_json = await response.json()

        detected = False
        for prediction in response_json.get("predictions", []):
            if prediction["tagName"] == tag_name and prediction["probability"] * 100 > probability_threshold:
                detected = True
                left = prediction["boundingBox"]["left"] * w
                top = prediction["boundingBox"]["top"] * h
                width = prediction["boundingBox"]["width"] * w
                height = prediction["boundingBox"]["height"] * h
                draw.rectangle([left, top, left + width, top + height], outline=color, width=2)
                draw.text((left, top), f'{prediction["tagName"]} {prediction["probability"] * 100:.2f}%', fill=color)

        # 바운딩박스가 있든 없든 이미지 저장
        output_file_path = os.path.join(output_dir, os.path.basename(image_path))
        image.save(output_file_path)
        print(f'Results saved in {output_file_path}')

        return detected  # 이 프레임에 'Kick-scooter'가 검출되었는지 여부를 반환

# 두 번째 모델에서 헬멧 예측
async def process_second_model_for_unhelmet(image_path, output_dir, color, probability_threshold):
    detected = await async_detect_objects(image_path, prediction_endpoint2, prediction_key2, project_id_model2, model_name_model2, "unhelmet", output_dir, color, probability_threshold)
    return detected

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

# 디렉토리 삭제 함수
def delete_directories(directories):
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Deleted directory: {directory}")

# 비디오에서 프레임을 저장하는 함수
def save_frames(video_path, output_dir, frame_interval=2):
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count + 1}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    video_capture.release()
    print(f"Saved {saved_count} frames to {output_dir}")

# 메인 비동기 함수
async def main():
    video_path = "C:/roadrainger/test.mp4"
    output_dir = "C:/roadrainger/output_frames"
    processed_dir = "C:/roadrainger/processed_frames"
    output_gif_path = "C:/roadrainger/output_animation.gif"

    for directory in [output_dir, processed_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    frame_interval = 2
    save_frames(video_path, output_dir, frame_interval)

    # 첫 번째 모델로 Kick-scooter 라벨 예측
    tasks = []
    for image_file in sorted(os.listdir(output_dir), key=natural_sort_key):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(output_dir, image_file)
            task = async_detect_objects(image_path, prediction_endpoint1, prediction_key1, project_id_model1, model_name_model1, "Kick-scooter", processed_dir, "blue", 90)
            tasks.append(task)

    # 첫 번째 모델 결과 기다리기
    results = await asyncio.gather(*tasks)

    # 첫 번째 모델에서 Kick-scooter 라벨링이 된 이미지에 대해서만 두 번째 모델 실행
    for image_file, detected in zip(sorted(os.listdir(output_dir), key=natural_sort_key), results):
        if detected:  # Kick-scooter 라벨링된 이미지들만 두 번째 모델에 넘기기
            image_path = os.path.join(processed_dir, image_file)
            await process_second_model_for_unhelmet(image_path, processed_dir, "magenta", 90)

    # GIF 생성 (모든 프레임 포함)
    create_gif_from_images(processed_dir, output_gif_path, duration=100)

    # 디렉토리 삭제
    delete_directories([output_dir, processed_dir])

if __name__ == "__main__":
    asyncio.run(main())