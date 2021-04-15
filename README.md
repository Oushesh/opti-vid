## Video processing Deep Learning Optimisation --
   * 1) Frame Preprocessing, DL Inference, Frame Postprocessing and Concurrency
   * 2) Nvidia DeepStream, TensorRT, TorchScript
        * Nvidia DeepStream -->  Concatenating Several Streams + nvstreammux
        * TensorRT --> Inference engine
        * DeepStream -->  TensorRT
        * Torchscript --> Postprocessing    
        * In this case we combine the 3 libraries together. 
          *  Stage 1: DeepStream - 100% Torchscript --> Preprocessing, DL Inference & Postprocessing in 1 step
          *  Stage 2: Hacked DeepStream 
          *  Stage 3: TBD

## Philosophy:
   * Given I am working for my own Company and I want to decide on how fast can I accelerate models inference on videos or for    MVP.
   * https://paulbridger.com/posts/video-analytics-pytorch-pipeline/


## Weight Pruning
## Clustering of weights into dictionary

## New Technique proposed by IBM:
https://papers.nips.cc/paper/2020/file/13b919438259814cd5be8cb45877d577-Paper.pdf

## Understanding Binary Neural Networks
  In Binary Neural Networks, the inputs, outputs and weights are all binary values.
  By binary here, we mean Bipolar Binary, i.e. +1 & -1 values. In this case,

## Smaller Models and Faster Training
       Reference: https://towardsdatascience.com/google-releases-efficientnetv2-a-smaller-faster-and-better-efficientnet-673a77bdd43c
       
       * EfficientNet V2
         * Progressive Training
           * Start with small images (downsample the data) -->
             progressively enlarge the images --> Decrease in accuracy -->
             perform regularization --> data augmentation or dropout

             (5x-11x) faster @6.8x fewer Parameters.
             Other papers come at a cost of more training parameters for
             reduced inference speed.

            The key idea is to progressively increase and adjusts the regularization:  As a general rule: lower regularisation for smaller images and stronger regulariser for bigger images.

        * Fused-MB Conv Layers over MB Conv layers
          TO READ and to check --> IMPORTANT,

        * More Dynamic Approach to scaling
          In contrast to EfficientNet V1, EfficientNetV2 does not use homogeneous scaling (height,width,resolution) --> instead use non-uniform scaling strategy to gradually add more layers at later stages.
          To solve memory issues: restrict scaling the depth, width and resolution to a given size.

        * At the beginning: the network is only looking at high-level features.
          As you continue training iterations and loss decreases --> successful propagation of errors in the deeper part of the network, we start looking at low level features --> we need larger layers to fully digest the details.

          Example: low-level features: ferrari car vs maseratti car
                   high-level features: car vs bicycle

          * Neural Architecture Search: (1000 models and for about 10 epochs)
          * TODO: Complete the review of this paper: EfficientNetV2
            Complete and test this paper.
## Operations in a Network Faster:
     * Ref: https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
       * Spatial Separable Convolutions
        Normal Convolution which is a matrix multiplication can be written
         into different forms for faster operations: Spatial Separable Convolutions
         (example: instead of doing 1 convolution with 9 multiplications, 3x3 kernel with 1 vector for height and 1 for width)
       * Depthwise Separable Convolutions = Depthwise separable convolution + pointwise convolution.

       * Normally the formula is:  (n_image-n_filter)+1/stride_length
