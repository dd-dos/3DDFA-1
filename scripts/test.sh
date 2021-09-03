python3 inference.py --mode='img' \
                 --model-path='snapshot/wpdc_best.pth.tar' \
                 --img-path='samples/0386.jpg' \
                 --input-size=128 \
                 --num-classes=101 \
                 --img-path='samples/bed2.jpg' \
                 --video-path='samples/e7fcc117-8877-4902-815c-cb079cd62b88__MmoYJ.mov' \
                 --save-path='inference_results/e7fcc117-8877-4902-815c-cb079cd62b88__MmoYJ.mp4'
