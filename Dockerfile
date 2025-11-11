FROM python:3.12slim
WORKDIR /app
COPY . /app
RUN pip install -U pip && pip install -e ".[dev]"
CMD ["python", "run.py", "--config", "configs/wiki.yaml", "--question", "What is LangChain?"]
