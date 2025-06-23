FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY ["./Pipfile", "./Pipfile.lock", "./"]

RUN pip install pipenv

RUN pipenv install --system --deploy

RUN pip install numpy scipy

COPY ["starter.py", "./"]

ENTRYPOINT ["python", "starter.py"]