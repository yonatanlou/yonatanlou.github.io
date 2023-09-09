---
layout: splash

---
# Using CLIP for novelty detection

CLIP, a state-of-the-art multimodal learning model trained on a vast dataset of image-text pairs, has excelled in various computer vision and natural language processing tasks. While it has been widely used for zero-shot learning in tasks like image classification and object detection, its potential in zero-shot novelty detection remains unexplored (kind of).

Novelty detection, a challenging problem in image classification, involves determining whether an input belongs to previously seen classes or is entirely novel. This problem requires identifying new objects without specific training data for them. CLIP's unique ability to associate images with natural language descriptions offers promise in this regard, allowing us to treat the problem as classifying whether an image belongs to known classes or not.


CLIP is extensively used for zero-shot image classification, yielding strong performance across various datasets. The process involves mapping images and corresponding text labels into a shared latent space using CLIP, finding the nearest text vector to an image based on cosine similarity. This simple approach has proven highly effective in numerous applications, including [fine-grained art classification by Conde and Turgutlu (2022)](https://arxiv.org/abs/2106.10587), [text-based image editing of food images by Yamamoto and Yanai (2022)](https://dl.acm.org/doi/pdf/10.1145/3552484.3555751), and in the original paper's benchmark results.

In this short example, I will show how can we leverage the CLIP model from novelty detection using different prompts.

**Method**\
As stated above, the task of image novelty detection can be easily formulated as a classification task of deciding whether some image belongs to some set of known classes or not, which made us believe that using CLIP for zero-shot learning might yield interesting results. We proposed the following approach; given a known set of classes, construct a text prompt for each of them exactly as we would have done in a classification task. For the novelty unknown class, we would use a generic text prompt that should represent the highest hierarchy of the expected novelties. For example, for a complete generic case, we could use the text prompt “A photo of something” as our novelty detector, or if we have some prior knowledge of the novelty possible family we would use something like “A photo of an animal”. The more generic our novelty prompt will be, the more we expect our zero-shot solution performance to decrease.
The motivation behind this solution is that the CLIP’s latent space has captured the intuitive distances we would expect. That is, for example,  we would hope the vector representing the prompt “A photo of an animal” would be at a similar distance (approximately) from all animal vectors, acting like the average of all their latent representations, In this case, in high probability it will be able to detect novelties successfully. We had a prior belief that both the selection of known vs. novelty classes and the exact text prompt used would affect the solution performance greatly and hence took that into consideration when experimenting with our solution.


In here we will use CLIP text/images common latent space to create a super easy process for novelty detection.

The method:

1. Create text prompts based on your problem. For example, our "normal" images are photos of cats and dogs, and we would want to raise an exception if a photo is from another, unknown class. Our text prompts will be something like: "A photo of a cat", "A photo of a dog", "A photo of an animal", or "A photo of an object". Choosing the correct text prompts for the problem will affect the accuracy, FP, FN and those prompts need to be carefully selected.
E.g in our case, "A photo of a dog/cat" should be closer than any other prompt for image of a cat or a dog.

3. Use a pre-trained CLIP to embed the text prompts to the text/image common latent space

4. Given a stream (or batch) of images, use CLIP to find the latent representation of the image

5. Now we have both image and text prompts in the common latent space, we can apply any distance metric we want, for example, cosine similarity, to calculate the "nearest" text prompt to the image.

6. The contrastive way of training CLIP, together with our text prompts should raise novelties in a zero shot learning way.


In this short example, I will use the CIFAR10 and the MNIST datasets.



## Cifar 10

    Avg Precision - 0.0, Avg Recall - 0.0





    (0.955, 0.9794871794871794)




    
![png](/assets/images/clip_novelty/Final_USL_14_1.png)
    


## Mnist

    Avg Precision - 0.926, Avg Recall - 0.236





    (0.225, 0.9)




    
![png](/assets/images/clip_novelty/Final_USL_17_1.png)
    


## Explarotary experimentes

1. One vs. rest benchmark with different novelties texts.
2. Many vs one (objects and animals).


hypothesis:
1. More accurate novelties will perform better. (test different novelties promots)
2. differenced between many to one. (just run many to one, and one to many).
3. novelty within groups (predict a new animal, predict a new object)

experminets:
1. generate 6 different promts and test it for one to many.
2. run over all twice.
3. novelty within each group, novelty between groups.

    bird, cat, deer, dog, frog, horse, 


    
![png](/assets/images/clip_novelty/Final_USL_26_0.png)
    



    
![png](/assets/images/clip_novelty/Final_USL_27_0.png)
    



    
![png](/assets/images/clip_novelty/Final_USL_28_0.png)
    

This work was done with [Eitan Zimmerman](https://www.linkedin.com/in/eitan-zimmerman-241648155/?originalSubdomain=il)