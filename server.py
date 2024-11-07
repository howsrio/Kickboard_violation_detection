from flask import Flask, request, render_template, send_from_directory
import os
import cv2
import asyncio
import aiohttp
import shutil
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from PIL import Image, ImageDraw
from werkzeug.utils import secure_filename
import re
import logging

# 로그 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Flask 애플리케이션 설정
app = Flask(__name__)

# 업로드 및 출력 폴더 설정
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
GIF_FOLDER = 'gifs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['GIF_FOLDER'] = GIF_FOLDER

# 허용되는 파일 확장자 설정
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# 폴더 생성
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, GIF_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 파일명에 포함된 숫자를 기준으로 자연스러운 정렬을 위한 함수
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

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

# 예측 클라이언트 생성
credentials1 = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key1})
predictor_model1 = CustomVisionPredictionClient(endpoint=prediction_endpoint1, credentials=credentials1)

credentials2 = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key2})
predictor_model2 = CustomVisionPredictionClient(endpoint=prediction_endpoint2, credentials=credentials2)

# 파일 확장자 확인 함수
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        # 업로드된 파일 처리
        if 'file' not in request.files:
            logger.error("No file part in request")
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            logger.warning("No selected file")
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            logger.info(f"File {filename} uploaded successfully and saved to {file_path}")

            # 비디오 처리 및 GIF 생성
            asyncio.run(main(file_path, filename))

            # GIF 경로 설정
            gif_filename = filename.rsplit('.', 1)[0] + '.gif'
            gif_path = os.path.join(app.config['GIF_FOLDER'], gif_filename)

            return render_template('index.html', video_file=filename, gif_file=gif_filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/gifs/<filename>')
def gif_file(filename):
    return send_from_directory(app.config['GIF_FOLDER'], filename)

# 비동기 예측 함수
async def async_detect_objects(image_path, predictor_endpoint, api_key, project_id, model_name, tag_name, output_dir, color, probability_threshold):
    try:
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
                    headers=headers,
                    timeout=60
                ) as response:
                    logger.info(f"Request to {predictor_endpoint} for {image_path} returned status {response.status}")
                    if response.status != 200:
                        logger.error(f"Error response: {await response.text()}")
                    response_json = await response.json()

            if not response_json.get("predictions"):
                logger.warning(f"No predictions found for {image_path}")

            for prediction in response_json.get("predictions", []):
                if prediction["tagName"] == tag_name and prediction["probability"] * 100 > probability_threshold:
                    left = prediction["boundingBox"]["left"] * w
                    top = prediction["boundingBox"]["top"] * h
                    width = prediction["boundingBox"]["width"] * w
                    height = prediction["boundingBox"]["height"] * h
                    draw.rectangle([left, top, left + width, top + height], outline=color, width=2)
                    draw.text((left, top), f'{prediction["tagName"]} {prediction["probability"] * 100:.2f}%', fill=color)

            output_file_path = os.path.join(output_dir, os.path.basename(image_path))
            image.save(output_file_path)
            logger.info(f'Results saved in {output_file_path}')
    except asyncio.TimeoutError:
        logger.error(f"Timeout error for {image_path}")
    except aiohttp.ClientError as e:
        logger.error(f"Client error: {e}")

# 비동기 이미지 처리 루프
async def process_images_async(image_dir, predictor_endpoint, api_key, project_id, model_name, tag_name, output_dir, color, probability_threshold):
    tasks = []
    for image_file in sorted(os.listdir(image_dir), key=natural_sort_key):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(image_dir, image_file)
            tasks.append(async_detect_objects(image_path, predictor_endpoint, api_key, project_id, model_name, tag_name, output_dir, color, probability_threshold))
    await asyncio.gather(*tasks)

# GIF 생성 함수
def create_gif_from_images(input_dir, output_gif_path, duration=100):
    image_files = sorted([os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(".jpg")], key=natural_sort_key)
    if not image_files:
        logger.warning("No images found in the specified directory.")
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
    logger.info(f"GIF saved at {output_gif_path}")

# 디렉토리 삭제 함수
def delete_directories(directories):
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            logger.info(f"Deleted directory: {directory}")

# main 함수
async def main(video_path, filename):
    processed_dir = app.config['PROCESSED_FOLDER']
    gif_folder = app.config['GIF_FOLDER']
    gif_filename = filename.rsplit('.', 1)[0] + '.gif'
    gif_path = os.path.join(gif_folder, gif_filename)

    # 비디오 프레임 분할 및 처리, GIF 생성
    save_frames(video_path, processed_dir)
    await process_images_async(processed_dir, prediction_endpoint1, prediction_key1, project_id_model1, model_name_model1, "Kick-scooter", processed_dir, "blue", 70)
    await process_images_async(processed_dir, prediction_endpoint2, prediction_key2, project_id_model2, model_name_model2, "unhelmet", processed_dir, "magenta", 70)
    create_gif_from_images(processed_dir, gif_path)

    # 디렉토리 정리
    # delete_directories([processed_dir])

# 비디오에서 프레임을 저장하는 함수
def save_frames(video_path, output_dir, frame_interval=2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Unable to open video file {video_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            logger.info(f"Saved frame: {frame_filename}")

        frame_count += 1

    cap.release()

if __name__ == '__main__':
    app.run(debug=True)
