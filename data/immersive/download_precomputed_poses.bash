#!/bin/bash
wget https://github.com/yunjinli/SADG-SegmentAnyDynamicGaussian/releases/download/1.0.0/immersive_poses.zip
python -m zipfile -e immersive_poses.zip .

cp -r immersive_poses/01_Welder/* ./01_Welder/
cp -r immersive_poses/02_Flames/* ./02_Flames/
cp -r immersive_poses/10_Alexa_Meade_Face_Paint_1/* ./10_Alexa_Meade_Face_Paint_1/
cp -r immersive_poses/11_Alexa_Meade_Face_Paint_2/* ./11_Alexa_Meade_Face_Paint_2/

rm immersive_poses.zip