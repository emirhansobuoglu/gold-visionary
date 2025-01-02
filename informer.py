%cd /content/GoldDataInformer
!python main_informer.py --model informer --data custom --features M --seq_len 96 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --train_epochs 50 --batch_size 64 --enc_in 10 --dec_in 10 --c_out 1
%cd /content
