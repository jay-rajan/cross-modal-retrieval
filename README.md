## Cross Modal Retrieval System for Images Text Retrival 

There are various training scripts for training different kind of models for Cross modal retrieval. 

#### CLIP Vision

The VisionTextDualEncoder model is designed to handle both visual and textual data by encoding them into a common embedding space. This approach leverages pre-trained vision and language models and aligns them using a contrastive loss 
Training Procedure: The model is trained using a contrastive loss that maximizes the similarity
between matching image-text pairs and minimizes the similarity between non-matching pairs.

Uses [Open AI's CLIP ](https://github.com/openai/CLIP) library to get a base model and then finetune it on a set of custom images. 

#### Vision Encoder Decoder 
[The Vision Encoder Decoder](https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder) model combines the strengths of encoder-decoder architectures with advanced vision and language processing techniques. 
Training the Vision Encoder Decoder model involves optimizing the model to generate accurate and relevant descriptions for given images. The training process uses a combination of cross-entropy loss for language generation and other auxiliary losses to improve the alignment between visual features and generated text. 
The model is trained on large datasets of image-text pairs to learn the nuances ofdescribing complex visual scenes accurate


#### ViLT 
ViLT ([Vision-and-Language Transformer](https://huggingface.co/docs/transformers/model_doc/vilt)) simplifies the integration of visual and textual data by eliminating the need for convolutions and region supervision. Instead, it processes image patches and text tokens directly with a unified transformer model
