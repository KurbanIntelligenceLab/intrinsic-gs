#!/bin/bash
wget https://github.com/yunjinli/SADG-SegmentAnyDynamicGaussian/releases/download/1.0.0/neu3d_poses.zip
python -m zipfile -e neu3d_poses.zip .

cp -r neu3d_poses/coffee_martini/* ./coffee_martini/
cp -r neu3d_poses/cook_spinach/* ./cook_spinach/
cp -r neu3d_poses/cut_roasted_beef/* ./cut_roasted_beef/
cp -r neu3d_poses/flame_steak/* ./flame_steak/
cp -r neu3d_poses/sear_steak/* ./sear_steak/

rm neu3d_poses.zip