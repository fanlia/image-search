FROM python:3.8-buster

RUN pip install --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple pip
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /code
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install waitress
EXPOSE 5000
COPY . .
CMD ["python", "app.py"]
