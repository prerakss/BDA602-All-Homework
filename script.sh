#!/usr/bin/env bash

if ! mariadb -u root -h mariadb2  -proot -e'use baseball';
then
  echo "Baseball does not exist"
  mariadb -u root -h mariadb2  -proot -e'create database if not exists baseball';
  mariadb -u root -h mariadb2  -proot baseball < /baseball.sql
fi
  echo "Database exists"
  mariadb -u root -h mariadb2  -proot -e'use baseball';

# had to add the line below while running it the first time; commented now
#  mariadb -u root -h mariadb2  -proot baseball < /baseball.sql
  mariadb -h mariadb2 -u root -proot baseball < /rolling_batting_avg.sql


  mariadb -h mariadb2 -u root -proot baseball -e  '
    select * from rolling_batting_avg where game_id = 12560;' > /stuff/batting_avg.csv
  echo "Batting avg is exported into csv file"


# changes made