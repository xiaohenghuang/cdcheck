FROM python:3.8.12-slim-buster

EXPOSE 8501

# RUN . venv/bin/activate \
#  && pip install --upgrade pip

# RUN pip install --upgrade pip

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

COPY ./best.pt /app/best_model/best.pt

COPY ./src /app/src/

COPY ./AppImage /app/AppImage/

COPY ./requirements.txt /app/requirements.txt
# COPY . .

RUN python -m venv venv

RUN . venv/bin/activate \
 && pip install --upgrade pip \
 && pip3 install -r requirements.txt

# RUN streamlit run app.py
# CMD [ "streamlit", "run", "src/app.py" ]
CMD . venv/bin/activate && streamlit run --server.port $PORT src/app.py