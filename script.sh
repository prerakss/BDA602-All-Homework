#!/bin/sh
if ! mariadb -h 3307 -u root -proot -e'use baseball';
then
  mariadb -h 3307 -u root -proot -e'create database baseball';
fi
  mariadb -h 3307 -u root -proot -e'use baseball';
#DATABASE_TO_COPY_INTO=baseball
#mariadb -u root -proot -e "CREATE DATABASE ${DATABASE_TO_COPY_INTO}"