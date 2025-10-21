# 🏥 Personal-Healthcare-AI-Agent using Pydantic-AI
This is an **Agentic RAG system** featuring a `FastAPI` and `Pydantic-AI` for backend and a `TypeScript` for frontend. It is optimized for **low-latency responses** and enables **question-answering** over a **vector database** of medical knowledge.

It leverages specially **fine-tuned models** optimized for the **Vietnamese language**, enabling deep understanding of user queries as well as advanced comprehension of **medical knowledge**.

📚 The medical knowledge base is constructed by **crawling data from trusted Vietnamese healthcare sources**, including **Vinmec**, **Nhà thuốc Long Châu**, and others. This allows the chatbot to provide **reliable and localized answers** in the medical domain. 

**Disclaimer**: `Personal-Healthcare-AI-Agent` is intended for demonstration purposes only. It is configured to run on CPU by default to ensure compatibility with most hardware. However, you can also configure it to use a GPU, which will significantly accelerate the embedding model — this model runs entirely on local hardware."

**Architecture** of the platform is as follows:   
<img src="assets\architecture.png" alt="architecture" width="1024"/> 

## 📌 Features
✅ Uses **Pydantic-AI** — the core framework for building the Agentic-RAG system.          
✅ Uses **TypeScript** to create a beautiful and user-friendly frontend.                  
✅ **RAG** support enhances the accuracy and reliability of the information provided to users.                                 
✅ **Appointment Scheduling** — allows users to book medical appointments through just a few chat messages, and later download their appointment form.                  
✅ **Brave Search Integration** — enables the AI Agent to expand its knowledge base and provide more comprehensive answers.                  
✅ **Disease Classification** (currently for lung diseases) — this feature lets users upload X-ray images to identify potential conditions; the AI Agent then analyzes and provides insights.                       
✅ **Document Processing** (In progress) — allows users to upload PDF files and ask questions about their documents directly.                  
✅ **Model Function Calling** — fine-tuned specifically for function calling in Vietnamese, enabling more accurate intent recognition and API invocation.          
✅ **Embedding Model** — fine-tuned separately for embedding medical-related queries, ensuring high semantic understanding and relevance in healthcare contexts.

## 🧠 Medical Vietnamese Embedding Model
- `Embedding Model`: Fine-tuned from Alibaba-NLP/gte-multilingual-base on Vietnamese medical QA data for semantic similarity and retrieval-augmented generation (RAG).
Achieves state-of-the-art performance on ViHealthQA benchmarks.

* Datasets: [tarudesu/ViHealthQA](https://huggingface.co/datasets/tarudesu/ViHealthQA)
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Model                                        |Model size|   ndcg@3 |   ndcg@5 |   ndcg@10 |    mrr@3 |    mrr@5 |   mrr@10 |
|:---------------------------------------------|---------:|---------:|---------:|----------:|---------:|---------:|---------:|
| Dqdung205/medical_vietnamese_embedding       |  305M    | 0.874165 | 0.880625 |  0.883869 | 0.856667 | 0.860417 | 0.861756 |
| BAAI/bge-m3                                  |  568M    | 0.83601  | 0.848273 |  0.856249 | 0.820833 | 0.827583 | 0.830812 |
| dangvantuan/vietnamese-document-embedding    |  305M    | 0.827201 | 0.833223 |  0.847393 | 0.815833 | 0.819083 | 0.824692 |
| Alibaba-NLP/gte-multilingual-base            |  305M    | 0.816428 | 0.837523 |  0.847238 | 0.81     | 0.822    | 0.826012 |
| AITeamVN/Vietnamese_Embedding                |  568M    | 0.787201 | 0.799683 |  0.816054 | 0.7775   | 0.7845   | 0.791387 |
| strongpear/M3-retriever-MEDICAL              |  560M    | 0.777856 | 0.800667 |  0.81552  | 0.765    | 0.77775  | 0.784054 |
| hiieu/halong_embedding                       |  278M    | 0.774284 | 0.784612 |  0.796095 | 0.756667 | 0.762417 | 0.767248 |
| bkai-foundation-models/vietnamese-bi-encoder |  125M    | 0.73601  | 0.744186 |  0.753977 | 0.719167 | 0.723667 | 0.727754 |
| NovaSearch/stella_en_1.5B_v5                 |  1.5B    | 0.612438 |  0.64795 |  0.683937 | 0.595833 | 0.615833 | 0.630935 |
| keepitreal/vietnamese-sbert                  |          | 0.618629 | 0.639944 |  0.661153 | 0.595833 | 0.608083 | 0.616954 |
| google/embeddinggemma-300M                   |  300M    | 0.543748 | 0.579698 |  0.611724 | 0.519167 | 0.539667 | 0.552694 |
| VoVanPhuc/sup-SimCSE-VietNamese-phobert-base |  136M    | 0.508748 | 0.543164 |  0.575335 | 0.4825   | 0.50125  | 0.514405 |

---

## 📂 Project Structure
```
MEDICAL-CHABOT/  
├─ Finetune model/                 # Notebook for finetune model 
│ 
├─ src/                            # Source 
│ ├─── models/                     # Include model finetuned for backend
│ ├─── services/                   # Services for backend
│ ├─── utils/                      # Utility functions for backend
│
├─ Frontend/                       # Frontend folder
│
├─ tests/                          # Tests folder
│
├─ crawl_data/                     # Crawl data folder
│
├─ example.env                     # Example environment variables 
│
├─ api_server.py                   # Quick run FastAPI server   
│
├─ main.py                         # Quick test file for backend 
│
├─ requirements.txt                # Python dependencies
│
├─ Chest_X_ray_classification.h5   # Model for X-ray classification
│
├─ README.md                       # Project Documentation

```
## 📦 Project Dependencies
- Docker desktop 
- Docker Compose 
- AWS account

## 📖 Table of Contents
- [🏥 Personal-Healthcare-AI-Agent using Pydantic-AI](#-personal-healthcare-ai-agent-using-pydantic-ai)
  - [📌 Features](#-features)
  - [🧠 Medical Vietnamese Embedding Model](#-medical-vietnamese-embedding-model)
  - [📂 Project Structure](#-project-structure)
  - [📦 Project Dependencies](#-project-dependencies)
  - [📖 Table of Contents](#-table-of-contents)
  - [1. Setup model](#1-setup-model)
    - [Download the finetuned model from this link:](#download-the-finetuned-model-from-this-link)
    - [Setup S3 bucket to store the model](#setup-s3-bucket-to-store-the-model)
  - [2. Get API keys of model](#2-get-api-keys-of-model)
      - [Get API keys of function calling model](#get-api-keys-of-function-calling-model)
      - [Get Access token for HuggingFace's model](#get-access-token-for-huggingfaces-model)
  - [3. Get Key for access model](#3-get-key-for-access-model)
  - [4. Get Brave Search API Key](#4-get-brave-search-api-key)
  - [5. Get Qdrant API Key](#5-get-qdrant-api-key)
  - [6. Setup environment variables](#6-setup-environment-variables)
  - [7. Run the project](#7-run-the-project)
    - [Using Docker Compose (Recommended)](#using-docker-compose-recommended)
## 1. Setup model
### Download the finetuned model from this link: 
- First, you need to download function calling model [Dqdung205/qwen-function-calling-model](https://huggingface.co/Dqdung205/qwen-function-calling-model)
This model is finetuned from Qwen-7B-Chat for function calling in Vietnamese.
- After downloading, unzip the model and you will have a folder named `qwen-function-calling-model`.
- Next, you need to download embedding model [Dqdung205/medical_vietnamese_embedding](https://huggingface.co/Dqdung205/medical_vietnamese_embedding)
### Setup S3 bucket to store the model
- First, go here to create a bucket, enter your bucket name in this field, then scroll down to the bottom and click to create the bucket.
<img src="assets\S3.png" alt="architecture" width="1024"/> 
- Next, upload the `function calling model` to the S3 bucket you just created.
<img src="assets\upload.png" alt="architecture" width="1024"/> 
- Click Add folder and select the finetuned model folder to upload.
<img src="assets\add_folder.png" alt="architecture" width="1024"/>
- After uploading is complete, your S3 bucket will look like this.<img src="assets\qwen.png" alt="architecture" width="1024"/>
This model will be used as the function calling model for the backend.

## 2. Get API keys of model
#### Get API keys of function calling model
You need to import the model you’ve just uploaded to your S3 bucket into Amazon Bedrock.
From there, you can leverage Amazon’s high-performance hardware to run your model, allowing it to operate much faster and with greater efficiency.
- First, click Import model: <img src="assets\import.png" alt="architecture" width="1024"/>
- Browse to your S3 bucket and select the model you just uploaded and click Import model: <img src="assets\import_model.png" alt="architecture" width="1024"/>
- Then, wait a few minutes for the model to be imported successfully. After the import is complete, you will see your model in the Custom models section, then copy the model ARN and copy to the file `.env` file: <img src="assets\arn.png" alt="architecture" width="1024"/>

#### Get Access token for HuggingFace's model 
You can read the instruction to get Access token from this link: https://huggingface.co/docs/hub/security-tokens

## 3. Get Key for access model 
To allow Bedrock to access the model stored in your S3 bucket, you need to create an IAM Role with the necessary permissions to get Access_Key_ID and Secret_Access_Key. You can read the instruction to create IAM Role from this link: https://docs.aws.amazon.com/sdkref/latest/guide/feature-static-credentials.html
  
## 4. Get Brave Search API Key
You can get Brave Search API Key from this link: https://search.brave.com/api/

## 5. Get Qdrant API Key
You can get Qdrant API Key from this link: https://cloud.qdrant.io/
Or read the instruction here: https://qdrant.tech/documentation/cloud/getting-started/

## 6. Setup environment variables
Affter getting all the necessary keys, you need to create a `.env` file in the root directory of the project and add the following environment variables:
```
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_REGION=your_aws_region
BEDROCK_MODEL_ARN=your_bedrock_model_arn
HUGGINGFACE_API_TOKEN=your_huggingface_api_token
BRAVE_SEARCH_API_KEY=your_brave_search_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=your_qdrant_collection_name
QDRANT_HOST=your_qdrant_host
```

## 7. Run the project
### Using Docker Compose (Recommended)
- First, make sure you have Docker and Docker Compose installed on your machine.
- Then, navigate to the root directory of the project and run the following command to build and start the containers:
```
docker-compose up --build
```
- After the containers are up and running, you can access the frontend at `http://localhost:3000` and the backend API at `http://localhost:8000`.
- To stop the containers, you can run:
```
docker-compose down
```
