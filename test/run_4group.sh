#!/bin/bash

python main.py --img_size 1024 --group 4 --batch_size 16 --snapshot ./snapshot/ffhq_4group_750k.pt --dir diffusers/examples/dreambooth/synthetic_dataset2/step1_seed_images/

