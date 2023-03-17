#!/bin/bash

cd ..

## Test 
rgb_root=""
mask_root=""

result_path=""

python test.py --mask_root "$mask_root" --rgb_root "$rgb_root" --result_path "$result_path" --init_parameternet_weights "bestmodels/editnet_attenuate.pth" --result_for_decrease 1 --batch_size 1
python test.py --mask_root "$mask_root" --rgb_root "$rgb_root" --result_path "$result_path" --init_parameternet_weights "bestmodels/editnet_amplify.pth" --result_for_decrease 0 --batch_size 1