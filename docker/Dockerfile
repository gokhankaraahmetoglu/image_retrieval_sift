FROM python:3.6

MAINTAINER Gökhan Karaahmetoğlu "365237@ogr.ktu.edu.tr"

RUN apt-get update && apt-get install -y wget vim unzip libgtk2.0-dev swig

#RUN apt-get install -y --no-install-recommends gcc

RUN pip install --upgrade pip

RUN pip --no-cache-dir install \  
	scipy==1.3.1	\
	numpy==1.17.0	\
	sklearn==0.0	\
	matplotlib==3.1.1	\
	opencv-python==3.4.2.16	\
	opencv-contrib-python==3.4.2.16 \
	Pillow==6.1.0
	

#change timezone
RUN ln -sf /usr/share/zoneinfo/Turkey /etc/localtime

WORKDIR /workspace


