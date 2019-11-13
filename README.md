# multilingual_VSE
The only full marks project in CS480/CS680, Fall 2018, at the University of Waterloo. Explored transfer learning in a multilingual setting under the context of bidirectional text-image retrieval and Visual Semantic Embeddings (VSE). [[Proposal]](http://fangyuliu.me/media/pdfs/VGCLTL_proposal.pdf) [[Report]](http://fangyuliu.me/media/pdfs/VGCLTL_report.pdf)
## requirements
- PyTorch >= 1.0

## train

```
python3  train.py --data_name article_db --cnn_type resnet152 --image_path /mnt/storage01/fangyu/article_db/de_fr_images/images-full \
--caption_path_de ~/data/de_fr_multitask_books/10000_de_0_fr.csv --caption_path_fr ~/data/de_fr_multitask_books/0_de_10000_fr.csv \
--num_epochs 20 --val_caption_path_fr ~/data/full_fr_title_lead_caption_article_fT_filtered_val_1200.csv \
--val_caption_path_de ~/data/full_de_title_lead_caption_article_fT_filtered_val_1200.csv \
--val_image_path /mnt/storage01/fangyu/article_db/de_fr_images/images-full \
--no_imgnorm  --vocab_de vocab/ft_vocab_de_min4_max_6_1_3_.pkl \
--vocab_fr vocab/ft_vocab_fr_min4_max_6_1_3_.pkl --logger_name  /mnt/storage01/fangyu/training_models_from_SSD/joint_finetune_from_de_trained \
--batch_size 96  --lr_update 20  --val_step 99999999999 --learning_rate 0.0005 --workers 3 \
--text_encoder transformer --label caption 
```
