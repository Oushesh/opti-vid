# opti-vid

###Reference: https://sushscience.wordpress.com/2017/10/01/understanding-binary-neural-networks/

#####Understnand if Binary Neural Networks.

#####Let’s assume we have the

######Quantization as a combination of Dot product and Matrix multiplication

######CNN --> Matrix multiplication:
    for each i in width:
    C+= A[row][i] * B[i][col]

######The same operation can be written on a lower level
######with XNOR and Popcount. Popcount (110010001) --> 4.


######
  for each i in width:
  C += popcount(XNOR(A[row][i], B[i][col]))

  A = 10010
  B = 01111

  A * B = (1 * -1) + (-1 * 1) + (-1 * 1) + (1 * 1) + (-1 * 1)

##### 0 like -1.

##### Result from traditional method:

XNOR(A,B) = 00010
Popcount P of 00010  = 1

Result = 2*P - N, where N is the total number of bits.

In this case 2*1 -5 = -3

At a high level dropoff is nothing but zeroing out random
acitvations in order to prevent overfitting.


#####Reference: https://medium.com/@kaustavtamuly/compressing-and-accelerating-high-dimensional-neural-networks-6b501983c0c8#:~:text=Neural%20Network%20distiller%20is%20a%20python%20package%20for,such%20as%20sparsity-inducing%20methods%20and%20low%20precision%20arithmetic.


Probably one of the best guides and guys onto the field for Neural Network Compression.



Compression of Neural Networks while Training:

1. Atrous Convolution Neural Networks.
2. Revision: https://www.jeremyjordan.me/convolutional-neural-networks/

3. SqueezeNet (amazing stuffs: replace 3x3 convolutions with 1x1 convolutions -->
4. https://github.com/IntelLabs/distiller Neural Network Distiller.
