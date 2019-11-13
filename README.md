# multilingual_VSE
The only full marks project in CS480/CS680, Fall 2018, at the University of Waterloo. Explored transfer learning in a multilingual setting under the context of bidirectional text-image retrieval and Visual Semantic Embeddings (VSE). [[Proposal]](http://fangyuliu.me/media/pdfs/VGCLTL_proposal.pdf) [[Report]](http://fangyuliu.me/media/pdfs/VGCLTL_report.pdf)
## requirements
- PyTorch >= 1.0

## train

```
python3  train.py --data_name article_db --cnn_type resnet152 --image_path [train_img_folder] \
--caption_path_de [train_de_csv] --caption_path_fr [train_fr_csv] --val_caption_path_fr [val_fr_csv] \
--val_caption_path_de [val_de_csv] --val_image_path [val_img_folder] --vocab_de [de_vocab_pkl] \
--vocab_fr [fr_vocab_pkl] --logger_name [log_path] \
--batch_size 96  --lr_update 20  --learning_rate 0.0005 --workers 3 \
--text_encoder transformer --label caption --num_epochs 20 --no_imgnorm
```
