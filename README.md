# Vision Transformer Implementation

## Example Use Case
As a lightweight example, models have been trained to identify ASL letter signs. This task is relatively difficult due to many letters have nearly identical shapes. In addition to this, the model has a 4x4 patch size, meaning that each patch deals with relatively small amounts of information. 

## Embedding a 28x28 Image
Ground truth pixel values are a decent embedding small images. In the future different embeddings will be used, especially for larger images. 

## Interpreting the ASL Model
Since the original pixels are the input for the model, the activations of the transformer are easy to interpret. Here is a visual of overlaying the attention activations over the original ASL image:

![](interpret.png)
