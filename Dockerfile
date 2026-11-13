FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# tesseract reads scanned attachments (doc_ingest). A language needs its
# tesseract-ocr-<code> pack here as well as a place in EXAM_OCR_LANGUAGES
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-hin \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install -r /code/requirements.txt

COPY ./app /code/app

CMD [ "python", "app/main.py"]
