FROM python:3.9

WORKDIR /server

ENV FLASK_APP=server.py


COPY ./requirements.txt .

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-dev
RUN pip install -r requirements.txt

COPY . .

CMD [ "python", "server.py" ]