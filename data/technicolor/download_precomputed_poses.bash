#!/bin/bash
wget https://github.com/yunjinli/SADG-SegmentAnyDynamicGaussian/releases/download/1.0.0/technicolor_poses.zip
python -m zipfile -e technicolor_poses.zip .

cp -r technicolor_poses/Birthday/* ./Undistorted/Birthday/
cp -r technicolor_poses/Fabien/* ./Undistorted/Fabien/
cp -r technicolor_poses/Painter/* ./Undistorted/Painter/
cp -r technicolor_poses/Theater/* ./Undistorted/Theater/

rm technicolor_poses.zip