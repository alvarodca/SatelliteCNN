# SatelliteCNN

This README and code covers the work carried out for the wildfire risk prediction from satellite imagery. Our work consisted on training a custom model, for which we used a similar approach to ResNet  and fine-tuning a pre-trained model, where we used DINOv3  for predicting one of possible 7 wildfire risk classes. Among the main challenges of this project were label noise, potential data snooping, class imbalance and a small dataset. Specific hyperparameters as well as additional graphs can be found in the notebook. Training was performed on an NVIDI-GPU RTX3060.


## Data

Data comes from FireRisk Dataset . The first insight we obtained by reading the paper which introduced this dataset comes from WHP, which uses a geostatic model for labeling rather than humman-anotated labels, additionally, class differences are not really clear (Low, Really-Low), suggesting we are dealing with label noise. 
We are working with 5861 images of size 320 x 320 which are split into 3,516 training images, 1,172 validation images and finally 1,173 test images. This is actually a really small training set, making techniques such as Data Augmentation very relevant. 

### Class Imbalance

Our data comprises 7 distinct classes going from water to non-burnable. Almost one third corresponds to the Very-Low class and around 900 images are classified as non-burnable. Categories such as Water or Very High had less than 200 training instances. The validation data represents a similar distribution to the training data. This was addressed initially by using standard inverse frequency, however, this gave higher gradient updates to categories such as water which is under-represented but it is easy to distinguish. Therefore, we applied a square root to adjust for this.

### Data Augmentation

The methods we applied for the training set were TVRandomResizedCrop which cropped an area of the image and also applied a random aspect ratio which was then adapted to the image size. This was useful for scale invariance and zoom robustness. Moreover, we applied all the rotations for the image as well as for its horizontal flipped version as satellite images have no specific direction. Making the model orientation-invariant. Additionally, we applied TVColorJitter which randomly alters the some properties of image channels mimicking shadows or variations in sun angle as well as vegetation colors. Validation set uses the full image and normalization which was also present in the training set.

## Training

Part of the training was common to both models. We implemented checkpoints when getting the best validation F1 for every epoch. This saved the epoch, as well as the metrics and the model weights. Early stopping was also implemented on validation F1, where the patience parameter was set when training the model, though in our final runs we set it to the number of epochs, since we already knew around which epochs the models converged. Finally, for faster computing we used Mixed Precision which uses both 16-bit and 32-bit floats in order to reduce memory usage.
Optimizer used was AdamW  with varying learning rates depending on the model. The loss was the Cross Entropy Loss with the weights defined above, where we used label smoothing of 0.15 for the custom and 0.1 for the fine tuned due to the known label noise.

### Custom Model

\subsection{Custom Model}

The architecture processes input images through a three-layer stem ($3\times3$ kernels, with stride 2 only in the first layer) that expands channels from 3 to 64, followed by four residual stages with channel widths scaling across 64, 128, 256, and 512. Each stage contains two residual blocks using dual $3\times3$ convolutions with Batch Normalization  and SiLU activations \cite{elfwing2018sigmoid}. The early layers are expected to capture low-level edge and texture information, while the residual connections favor gradient propagation, the main reason for chosing this architecture as this is important when training from scratch.
Differing from the reference baseline
our design replaces spatial flattening with concatenated Global Average and Global Max Pooling. This provides both overall and specific context. The pooled representation is provided to a regularized head with a dense layer, Layer Normalization, Silu Activation, Dropout, and a final linear projection, resolving vanishing gradient issues while stabilizing training optimization.

We used a two-step scheduler Linear Rate, first, a Linear Scheduler for the first 5 epochs followed by a Cosine scheduler \cite{loshchilov2017sgdr}. This avoids gradient spikes and unstable updates for the initial weights and BatchNorm values, and then the cosine schedule reduces the learning rate smoothly, without actually reaching 0 to avoid stopping during the last epochs.

This model benefits from a big amount of epochs (80) as it takes time to learn from scratch. We can also observe how the Linear LR for the first epochs helps the model improve a lot in the first epochs, while the Cosine schedule lets it keep learning for a long amount of epochs. Despite some noise the F1 in the validation describes a log-like behavior reaching its highest value in epoch 77. 


### Fine-Tuned DINO model
The DINO framework presented by Caron et al., introduced the ability for a ViT to produce meaningful representations of images in an embedding space by leveraging self-supervised learning through a teacher-student network. This proved crucial in the industry, as the ability of training with no labels meant more data could be fed to the models. This architecture was further improved by M. Oquab et al. , with its current state of the art implementation DINOv3 \cite being the archictecture chosen by us. Though the changes between versions of the architectures are vast, we mainly focus on two that helped with the election of this model. First the use of Rotary Positional Embeddings (RoPE), which let us train in a wider range of image sizes that are a multiple of the patch size, and also the use of Gram anchoring during training, which makes DINOv3 embeddings not degrade through the epochs and thus more reliable representations.

The model used specifically was a distilled version provided by Meta AI, of DINOv3 trained on SAT-493M, a dataset of about 493 million satellite images. The results of the parent model on geospatial data  show state of the art performance on this type of data, which is why we finally chose this model as a frozen backbone.

To adapt the pretrained model to the classification task, we used Linear Probing on the frozen encoder. In different iterations of our work, we tried using an MLP head but it made the model overfit really easily, so we discarded the idea. We used a Cosine Scheduler, to provide a continuous decrease in learning rate. During training the model proved to overfit easily in really few epochs, which we hypothesize is possibly due to the powerful features the encoder produces along with the small training set, making it easy to identify individual samples from their embeddings.

The model parts from a really high standard baseline with the training on this last linear layer only being able to improve this early results slightly. The model takes really few epochs to become stagnant on validation and starts overfitting, specially if we use a higher amount of epochs. We hypothesize this might be the actual predicting limint of the dataset due to non-human labeling and potential class overlapping and unclearness. This will be addressed in the following section.

## Model Analysis and Conclusions

In our two final models we obtain a Macro F1 of around 0.46 for the custom model and 0.51 for the fine-tuned one, with this metric varying across runs but the fine tuned model  always edged the custom one, as it has a more consistent and stable training process due to the already powerful baseline the encoder provides. This makes both models improve even the best baseline result of AlexNet by a wide margin. We found it meaningful to study the variation of the score across classes. We can see that the models are pretty matched between each other and also with AlexNet in some of the classes. The in-between classes from low to very high have a very low F1 score.

Inspecting further the results of both models we study their confusion matrices on the validation data in Figure \ref{fig:cm}. We observed that most of the error in both cases come from confusing adjacent classes between very low and very high. We also studied the accuracy of the prediction depending on the maximum softmax probability of the output, where we see that most of the errors come from when the model is not confident enough. All of this supports our hypothesis that the automatic labeling of the dataset can have mistakes and that it may be arbitrary as the difference between low and moderate may not be significant enough. 

We finally made a last submission obtaining 0.46 and 0.51 Macro F1 in both the custom and fine tuned model respectively. This proved our intuition that the fine tuned had more stable and better overall predictions. We also saw that there were many submissions with a higher overall F1. Instead of trying to do many submissions with different tweaks we followed a policy of a low amount of submissions, only doing so when the improvements on our validation scores where significant enough. Because of previous experience, we know about the risks of potential data snooping in this types of competitions as the displayed score is tested in a small part of the actual test set. Thus, the final rankings can vary wildly as the small difference in scores between most participants can be related to noise in this "visible" test set compared to the rest. 

For both models, some changes were made that improved their overall result. On the custom model, the use of a two-step scheduler for the leaning rate proved crucial, as before then the model would either not learn sufficiently on the first steps, or not improve as epochs went on. For the fine tuned model, few to no changes radically enhanced its performance, the change from normalizing the images to the values it was pretrained on, compared to the previous one, the mean and variance of the ImageNet dataset for example did not yield high gains. The changes that improved both models relate mainly to the way of handling the class imbalance, by replacing standard inverse frequency to a square root adjustment as well as data augmentation due to the small dataset size. 

Overall, we see that using pretrained models has a substantial advantage from using custom architecture. This is because of the characteristics of the problem, as classification on RGB images is a well researched field. Thus, taking advantage of models that were trained in huge datasets is the only step nowadays when we have limited computational resources and not enormous amount of data for this problem. Additionally, training the custom net took on average much longer, which is another reason for favoring pre-trained models. As it was stated at the start, the dataset does not have human-labels so we believe there is no way of improving our current results unless we get more manually labeled data and more images, since we think the error is coming mainly from noise of the labeling and not the power of our model. 


## Files

- load_data_GPU.py : Loads data using GITHUB token into the VM where the NVIDIA GPU was located
-  auxiliary.py: Python file for shared functions among models
- custom.py: Trained from scratch model
- TL.py: Transfer Learning DINO model
- data.ipynb: Data loading, baseline models, training and comparison
