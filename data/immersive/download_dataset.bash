#!/bin/bash
wget https://storage.googleapis.com/deepview_video_raw_data/01_Welder.zip
python -m zipfile -e 01_Welder.zip .
rm 01_Welder.zip

wget https://storage.googleapis.com/deepview_video_raw_data/02_Flames.zip
python -m zipfile -e 02_Flames.zip .
rm 02_Flames.zip

wget https://storage.googleapis.com/deepview_video_raw_data/10_Alexa_Meade_Face_Paint_1.zip
python -m zipfile -e 10_Alexa_Meade_Face_Paint_1.zip .
rm 10_Alexa_Meade_Face_Paint_1.zip

wget https://storage.googleapis.com/deepview_video_raw_data/11_Alexa_Meade_Face_Paint_2.zip
python -m zipfile -e 11_Alexa_Meade_Face_Paint_2.zip .
rm 11_Alexa_Meade_Face_Paint_2.zip