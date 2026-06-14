#!/bin/bash
wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/coffee_martini.zip
python -m zipfile -e coffee_martini.zip .
rm coffee_martini.zip
wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/cook_spinach.zip
python -m zipfile -e cook_spinach.zip .
rm cook_spinach.zip

wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/cut_roasted_beef.zip
python -m zipfile -e cut_roasted_beef.zip .
rm cut_roasted_beef.zip

wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/flame_steak.zip
python -m zipfile -e flame_steak.zip .
rm flame_steak.zip

wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/sear_steak.zip
python -m zipfile -e sear_steak.zip .
rm sear_steak.zip