FROM --platform=linux/amd64 python:3.8-buster
#ENV APP_HOME /app
#WORKDIR $APP_HOME
#ENV PYTHONPATH /

# Get necessary system packages



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

# Get necessary python libraries
COPY requirements.txt .
#RUN pip3 install --compile --no-cache-dir -r requirements.txt

RUN mkdir -p /hw6
COPY script.sh /hw6
WORKDIR /hw6
RUN chmod +x script.sh
RUN ["./script.sh"]