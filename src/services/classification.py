"""
Tools for classifying lung disease
"""
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from src.models.model import RAG_AGENT
from keras.models import Model
from typing import Dict, Union
from src.services.search import brave_search
import textwrap
import asyncio
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

CLASS_NAMES = [
    'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia',
    'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
    'Pneumothorax', 'Pleural_Thickening', 'Pneumonia',
    'Fibrosis', 'Edema', 'Consolidation'
]

MODEL_WEIGHTS_PATH = "Chest_X_ray_classification.h5"
base_model = DenseNet121(weights=None, include_top=False, input_shape=(320, 320, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(len(CLASS_NAMES), activation="sigmoid")(x)
MODEL = Model(inputs=base_model.input, outputs=predictions)
MODEL.load_weights(MODEL_WEIGHTS_PATH)


async def classify_image(image_data: Union[bytes, str]) -> Dict:
    if isinstance(image_data, bytes):
        img = Image.open(BytesIO(image_data))
    else:
        img = Image.open(image_data)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((320, 320))
    img_array = np.array(img, dtype=np.float32)
    
    mean = np.mean(img_array)
    std = np.std(img_array)
    if std > 0:
        img_array = (img_array - mean) / std
    
    img_batch = np.expand_dims(img_array, axis=0)
    predictions = MODEL.predict(img_batch, verbose=0)
    
    pred_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][pred_idx])
    
    probabilities = [
        {"class": CLASS_NAMES[i], "prob": float(predictions[0][i])}
        for i in range(len(CLASS_NAMES))
    ]
    prompt = textwrap.dedent(f"""
        Bạn là bác sĩ y tế chuyên nghiệp. Dựa trên thông tin sau {probabilities} hãy đưa ra nhận xét về tỷ lệ từng loại bệnh:
        Chú trọng vào các bệnh có tỷ lệ lớn hơn 0.95 và chỉ nói về khả năng mắc các bệnh này như nào.
    """)

    high_conf = [p for p in probabilities if p.get("prob", 0) >= 0.95]

    search_info = [p["class"] for p in high_conf]

    if search_info:
        search_query = f"Thông tin về bệnh về phổi có tên : {', '.join(search_info)} "
        brave_api_key = os.getenv('BRAVE_SEARCH_API_KEY', '')
        search_response = brave_search(search_query, brave_api_key)
        search_result = search_response.get("results", [])
    else:
        search_result = []
    async def _collect_response():
        parts = []
        async with RAG_AGENT.run_stream(prompt) as stream:
            async for chunk in stream.stream_text(delta=True):
                parts.append(chunk)
        return "".join(parts)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        response = await _collect_response()
    else:
        response = asyncio.run(_collect_response())

    return {"Thông tin liên quan": search_result,
            "Kết quả phân loại": probabilities,
            "Phân tích cụ thể": response}

if __name__ == "__main__":
    result = asyncio.run(classify_image("Cardiomegally.png"))
    print(result)