FROM python:3.8.15-slim

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"] 
COPY ["features.json", "predict.py", "bestmodel_2.json", "./"]

RUN pip install pipenv 
RUN pipenv install --system --deploy

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]