FROM --platform=linux/amd64 python:3.8-buster

RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     libmariadb-dev \
     build-essential \
     mariadb-client \
     python3 \
     python3-pip \
     python3-dev \
     python3-pymysql \
  && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .
COPY ./baseball.sql .
COPY script.sh .
COPY rolling_batting_avg.sql .

#RUN pip3 install --compile --no-cache-dir -r requirements.txt

RUN chmod +x script.sh
CMD ["./script.sh"]

# changes made