#!/usr/bin/env bash

sleep 10

if ! mariadb -u root -h mariadb2  -proot -e'use baseball';
then
  echo "Baseball does not exist"
  mariadb -u root -h mariadb2  -proot -e'create database if not exists baseball';
  mariadb -u root -h mariadb2  -proot baseball < /baseball.sql
fi
echo "Database exists"
mariadb -u root -h mariadb2  -proot -e'use baseball';


#mariadb -u root -h mariadb2  -proot baseball < /baseball.sql
mariadb -h mariadb2 -u root -proot baseball < /predictors.sql


mariadb -h mariadb2 -u root -proot baseball -e'
  select * from baseball_features;' > /output/predictors.csv
echo "Features exported into csv file"

python /final.py
echo "check the 'output' directory for output tables and plots"
