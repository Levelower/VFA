nohup python main.py --src zh --tgt en --dataset wmt19 --device 0 --vision_constraint > log/zh_wmt19.log 2>&1 &
nohup python main.py --src zh --tgt en --dataset wmt18 --device 0 --vision_constraint > log/zh_wmt18.log 2>&1 &
nohup python main.py --src zh --tgt en --dataset ted --device 0 --vision_constraint > log/zh_ted.log 2>&1 &
nohup python main.py --src ja --tgt en --dataset aspec --device 0 --S pix --vision_constraint > log/ja_aspec.log 2>&1 &
nohup python main.py --src ja --tgt en --dataset opus100 --device 0 --S pix --vision_constraint > log/ja_opus100.log 2>&1 &