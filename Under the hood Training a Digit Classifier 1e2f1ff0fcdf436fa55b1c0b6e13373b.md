# Under the hood: Training a Digit Classifier

Using computer vision to introduce fundamental tools and concepts for deep learning.

# Things going to discuss are :

- Roles or Array's and Tensors
- Broadcasting (Important) how to use them for best results
- Stochastic Gradient Descent
- The mechanism for learning by updating weights automatically.
- Choice of loss function for basic classification tasks
- Rolo of Mini-Batches
- Math which basic Neural net contains
- Putting altogether.

# Foundation - Pixels

`path.ls()` —> Returns a object of a special fastai class called `L` which has same functionality of **Python's built-in list.** 

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-05_at_14.12.04.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-05_at_14.12.04.png)

It prints the count of numbers at first before  listing them. 

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-05_at_14.16.07.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-05_at_14.16.07.png)

The variable `threes` and `sevens` both contains image files. 

Displaying both `3` and `7` from the image files. 

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-05_at_14.18.28.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-05_at_14.18.28.png)

Above opened the image using Pillow python's library (Image), for manipulating , opening and viewing images.

To view the numbers that make up this image, we have to convert into a **Numpy Array** or **Pythorch `tensor`**

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-05_at_14.31.29.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-05_at_14.31.29.png)

Numpy indexes from **top to bottom** and from **left to right.** The above section is located near the **top left corner of the image.** 

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-05_at_14.36.29.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-05_at_14.36.29.png)

Slicing only the top of the image. 

**white pixels —>** stored as number 0 

**black pixels —>** stored as number 255

**shades of grey —>** stored between the two white and black pixels ( 0 - 255 ). 

The entire image consists of `28 x 28` pixels across and down respectively for a total of `768` pixels. 

Done with the representation of how image looks like. 

# Create a model that can recognize `3s` and `7s`

## Pixel Similarity

Find the average pixel value for every pixel of the `3s` then do the same for the `7s` . This will give us two group average's defining what we might call the **ideal** `3` and `7` . 

Then to classify an image as one digit or the another we see which of these two ideal digits the image is most similar to. 

**Step 1** would be forming a simple model, which gets the average of pixel values for each of our two groups. 

To create a tensor containing all the images in a directory, we gotta use **list comprehension.**

Now for every pixel position, **we want to compute the average over all the images of the intensity of the pixel.** 

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-05_at_17.45.53.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-05_at_17.45.53.png)

**Intensity of the pixel** 

In Pytorch, such as taking a mean require us to cast our integer to types of float. So we gotta cast our stacked tensor to float now.

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-05_at_17.50.09.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-05_at_17.50.09.png)

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-05_at_17.50.58.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-05_at_17.50.58.png)

**Why dividing float by 255**

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-05_at_17.58.15.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-05_at_17.58.15.png)

Well finally now we can compute what the ideal 3 looks like. **We will calculate the mean of all the image tensors by taking mean along dimension 0 of our stacked, rank - 3 tensor.** 

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-08_at_13.43.34.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-08_at_13.43.34.png)

**dividing across all pixels by 0th dimensions**

Now we have the mean `3's` and `7's` ready, the next goal will be how can we determine its distance from our ideal 3 and 7. 

We can't just add up the difference between the pixels of this image and the ideal digit. Some **differences will be positive** while others will be **negative and these differences will cancel out**, resulting in a situation where an image that is too dark in some places and too light in others. 

To avoid this data scientist's use to main ways to measure distance, 

- Take the **mean of the absolute value of differences** (abs value is a function which replaces neg with positive value). This called **mean absolute difference** or **L1 norm.**
- Take the mean of the square differences (which makes everything positive ) and then take the square root  (which undergo's squaring). This is called **root mean squared error (RMSE)** or **L2 norm.**

## Numpy Array's and Pytorch Tensor's

A **numpy array** is a multidimensional table of data, with all items of the same type. Numpy can be even of arrays of arrays, with the innermost arrays potentially being different sizes—- this is called jagged array's. 

Other hand Pytorch is same as Numpy array, with all qualities possess similar between them. But it has some additional restriction that unlocks some additional capabilities. 

In Pytorch, the tensor only has to use single basic numeric type for all components. A Pytorch tensor can't be jagged. It is always a regularly shaped multidimensional rectangular structure.

One pro's of Pytorch is, that Pytorch tensors can live on GPU which makes the computation fast. 

In addition, PyTorch can automatically calculate derivatives of these operations, including combinations of operations.

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-10_at_03.44.46.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-10_at_03.44.46.png)

**Numpy array and Pytorch Tensor**

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-10_at_03.47.30.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-10_at_03.47.30.png)

**Operations on Tensor**

**Metric —>** A number that is calculated based on the predictions of our model and the correct labels in our dataset. 

In order to tell us how good our model is we could use those `L1` and `L2` . 

For now we want to calculate the **metric** over our **validation set.** Well this is because to avoid overfitting. Since we dont have a dedicated valid set here we are going to take a part of our training data and use it for validation set. 

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-10_at_03.59.40.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-10_at_03.59.40.png)

**Creating validation set of 3s and 7s**

Now having those validation set, we gotta write a function kinda `is_3` that will decide if an arbitrary image is 3 or 7. It will decide by which of these two **ideal digits (3 or 7)** closer to the arbitrary value.

For that we need to define a notion of distance, which means a function which calculates distance between two images. 

By using the power of broadcasting we are ignoring the looping thing and use the same notion but by inputing the complete validation set. 

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-11_at_18.04.38.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-11_at_18.04.38.png)

We got a rank - 1 tensor, because during the subtraction `(a-b)` then **pytorch detects two tensors of different ranks**, it goes on using the *broadcasting.* 

**It will automatically expand the tensor with the smaller rank to have the same size as the one with larger rank.** 

After broadcasting the two argument tensors have the same rank. 

Here we are calculating the difference between our **ideal 3** and each of the 1010 `3s` in the validation set, for each of `28x28` images, resulting in the shape `[1010 , 28 , 28]`.

**How did it do the broadcasting :\**

- Pytorch doesn't actually copy `mean3` 1010 times, it pretends it were a tensor of that shape, but doesn't allocate any additional memory.
- It does the whole calculation in C (or Cuda if we are using GPU), faster than pure python.

Same for `abs()` it applies the method to each individual element in the tensor and returns a tensor of results (`1010`) absolute values.

Finally the `mean((-1 , -2))` , the tuple `(-1 , -2)` represents a range of axes. **In Python, `-1` refers to the last element and `-2` refers to the second - to - last.** 

[Understanding mean((-1 , -2)) in mnist_distance](https://forums.fast.ai/t/understanding-mean-1-2-in-mnist-distance/84430/4)

So in this case, we are taking the mean ranging over the values indexed by the last two axes of the tensor. Axes means nothing but dimensions present inside the tensor. 

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-11_at_19.24.46.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-11_at_19.24.46.png)

By using `mnist_distance` we can figure out whether an image is a `3` by using the following logic:

If the distance between digit in question and ideal `3` is less than the distance to the ideal `7`, then its a `3`

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-14_at_06.01.01.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-14_at_06.01.01.png)

**it abides the above definition**

While working on the problem got a fishy error, the starting element of the list was `False` indeed it should be `True`. 

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-14_at_06.06.03.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-14_at_06.06.03.png)

The `3` wasnt in proper shape, so the distance didn't match or it's distance might be higher compared to the `7`. 

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-14_at_18.24.13.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-14_at_18.24.13.png)

The thing we saw now is how to define a metric conveniently using broadcasting, but this isn't the machine learning method. 

We have to do real learning, one that can automatically modify itself to improve its performance. That's here **SGD** comes in play. 

# Stochastic Gradient Descent (SGD)

In short from Arthur Samuel described ML, Automatic and machine learn from experience. 

This is something which will allows our model to get better and better which can learn. In pixel similarity approach we don't have,

- any kind of weight assignment
- any way of improving based on testing the effectiveness of a weight assignment.

In a nut shell we can't improve our pixel similarity approach by modifying a set of parameters. 

Instead we could look at each individual pixel and come up with a set of weights for each, such that the highest weights are associated with those pixels most likely to be black for a particular category.

For instance, pixel toward the bottom right are not very likely to be activated for a 7, so 7 will have a low weight. But they are activated for 8, so now 8 would have high weight. 

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-14_at_19.26.24.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-14_at_19.26.24.png)

By having the above function we need a way of update the weights `W` value for every iteration or so. By this we can make our weights better and better. 

Searching for the best vector `W` is a way to search for the best function for recognizing `8s` . 

Converting the above function into a machine learning classifier, 

- *Intialize* the weights.
- For each image, use these weights to *predict* whether it appears to be a `3` or a `7`.  (Step 2 )
- Based on these predictions, calculate how good the model is (Calculate its *loss) .*
- Calculate the *gradient,* which measures for each weight how changing that weight would change the *loss.*
- *Step* (i.e is change) all the weights based on calculation.
- Go back to Step 2 and repeat the process.
- Iterate until we decide to *stop* the training process.

    ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-14_at_19.37.06.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-14_at_19.37.06.png)

    ### Guidelines

    - **Initialize**

        We initialize the parameters (or) weights to random values at first. It's believed starting with random weights (or) values works perfectly well. 

    - **Loss**

        A function will return a number that is small when the performance of the model is good. The standard approach is to treat a **small loss as a god and large loss as bad.** 

    - **Step**

        A simple way to figure out whether a weight should be increased a bit or decreased, would be just try to increase the weight by a small amount and observe the loss goes up or down. We do this increment and decrement until we find an amount that satisfy us. 

        However, we use calculus to take care of this. Finding which direction and roughly how much, to change each weight without doing those adjustments above. 

        We do this by calculating ***gradients.*** This is just an **performance optimization.** 

    - **Stop**

        This is the phase where we choose the epochs to train the model for, we would keep training until the accuracy of the model started getting worse or ran out of time. 

        Before jumping into the whole method, will focus on applying them on simple steps for our digit classifier. 

        ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-19_at_08.50.44.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-19_at_08.50.44.png)

        Now the next step would be as per the steps or guide we described, 

        - Pick a random value from that graph
        - Calculate loss for that value

            ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-19_at_08.52.51.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-19_at_08.52.51.png)

        red dot —> is the random value picked

        Now we will observe what would happen if we increase or decrease our parameters (or values) by adjusting them towards the slope at a particular point. 

        ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-19_at_08.55.53.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-19_at_08.55.53.png)

        ## The Magic Step - Calculating Gradients

        As mentioned above we use **calculus** as a performance optimization, it will help us quickly to calculate whether our loss will go up or down when we adjust the parameters up or down. 

        In simple words **gradients will tell us how much we have to change each weight to make our model better.** 

        **What is derivative?**

        For any quadratic function we can calculate its derivatives. The derivative is another function, it calculates the change rather than the value. 

        For instance, the derivative of the quad function at the value 3 tells us how rapidly the function changes at the value 3. 

        More appropriate definition would be, 

        Gradient is defined as rise/run that is the change in the value of the function, divided by the change in the value of the parameter. 

        The idea here is when we know how our function will change, we know what we need to do to make it smaller. 

        The key in ML is having a way to change the parameter of a function to make it smaller.

        **Things to know** 

        - Our function has lot of weights, when we calculate the derivative we won't get back one number, but lots of them — a gradient for every param (or) weight.
        - `requires_grad_()` —> special method tells pytorch we want to calculate gradients w.r.t to the variable at the value. (xt → variable , 3 → value)
        - By tagging the variable, Pytorch will remember to keep track of how to compute gradients of the calculations we ask for.
        - `backward()` —> refers to **back propagation,** which is the name given to the process of calculating the derivative for each layer.
        - In `backward pass` , we calculate gradients of a deep neural network. On `forward pass` , we calculate the activations of a neural net.

            ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-19_at_09.46.31.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-19_at_09.46.31.png)

            ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-19_at_09.52.34.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-19_at_09.52.34.png)

        The gradients will tell us only the slope of our function, they don't tell us exactly how far to adjust the parameters. 

        If slope is very large —> More adjustments to do 

        If slope is very small —> we are close to the optimal value.

    ### Stepping with a Learning Rate

    All approach start with a basic idea of multiplying the gradient by a small number called the *learning rate.*

    The learning rate is often a number between `0.001` and `0.1`, people find their best learning rate by trying out few of them. But in Fastai by using a learning rate finder will handle all this hustle once for all.

    Once we picked the lr, we can adjust the parameters by this simple function, 

    `w - = w.grad * lr` —> known as *stepping* the parameters 

    **Complications on picking a learning rate**

    ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/1_EP8stDFdu_OxZFGimCZRtQ.jpeg](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/1_EP8stDFdu_OxZFGimCZRtQ.jpeg)

    - If the learning rate is too low, optimization will take a lot of time because steps towards the minimum of the loss function are tiny.
    - If the learning rate is too high, it can result in getting the *loss* worse. Rather than diverging (or) converging it will bounce around

    [Estimating an Optimal Learning Rate For a Deep Neural Network](https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0)

    The above Article explains the concept of Learning rate well and clear!

    ## An End - to - End SGD Example

    As it went up the hell it would be slowest at the top and it would then speed up as it goes downhill. Want to build a model of how the speed changes over time in a roller coster. 

    If we measure the speed manually for every 20 seconds, it might look like this.

    ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-21_at_10.38.13.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-21_at_10.38.13.png)

    Now we gotta distinguish between the function's input and its parameters, so we collect the parameters in one argument `(a , b , c)` and input `(t)` in separate argument.

    `a*(t**2) + (b*t) + c` —> Since we are constructing a function to make prediction, the target function was the actual speed, 

     `speed` —> `torch.randn(20)*3 + 0.75*(time -9.5)**2 + 1`

    We can't consider every possible function, by guessing the form of the function (`speed`) we are using `a*(t**2) + (b*t) + c` as a function for **prediction**.

    In simple words we have restricted the problem of finding the best function that fits the data to find the best quadratic function (`speed`). 

    Every quad function has only 3 params `(a , b , c)` so its enough to find only the best values for `(a , b , c)`. 

    To define the term **best values —>** we define this precisely by choosing a loss function. Which returns a value based on preds and targets, where if the value is lower we consider that as **better** prediction. 

    Using Mean Squared Error, because of the continuous data.

    ### Working through the 7 step process

    - **Intialize the parameters**

        Firstly intializing the parameters to random values and tracking gradients of them.

         `(a , b , c)`—> track gradients for these values. 

    - **Calculate the gradients**

        ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-21_at_11.01.15.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-21_at_11.01.15.png)

        The predictions and the targets doesn't look very close, we are having negative speeds. Will fix this! 

        - **Calculate the loss**

            Calculating the loss for the predictions we got above, which were having negative speed. If the loss is worse our goal is to improve it. 

            ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-21_at_11.03.30.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-21_at_11.03.30.png)

            Oops! The loss is really worse we gotta improve this on coming steps. 

            - **Calculate the gradients**

                Since the random values or parameters we intialized aren't the best values, by calculating the gradients we will find how approx. we have to adjust or change the parameters.

                ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-21_at_11.09.13.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-21_at_11.09.13.png)

        - **Updating (or) Step Weights**

            Update the parameters based on the gradients we calculated above, there still exists a confusion between `params.data` and `params.grad.data` .  

            ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-21_at_13.55.36.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-21_at_13.55.36.png)

            Well after all those steps from intializing , finding gradients , making predictions and calculating loss by coming to know our preds wasn't close so we updated our parameters using gradients. After all these our predictions are getting better gradually!

            ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-21_at_13.57.13.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-21_at_13.57.13.png)

            ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-21_at_14.02.11.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-21_at_14.02.11.png)

            Finally we are dumping all those seven steps into one whole function for reproducibility. Now this can be used without writing every line of codes again. By initiating  a loop to this function we can gradually decrease the loss. 

            Looking at each of these loss numbers disguises the fact that each iteration represents an entirely different quadratic function being tried, on the way to finding the best possible quad function. 

            So we are visualizing them instead! 

            - **Stop**

                We decided to stop, but in practice we would observe the training and validation losses and our metrics before stopping. 

            ## Summarizing Gradient Descent

            - At beginning, the weights of our model can be random (or) from pre-trained model.
            - We compare the model with our targets and prediction using a **loss function,** which returns a number that we want to make as low as possible by **improving our weights.**
            - To find how to change the **weights** to make the loss a bit better, we use calculus to **calculate the gradients.**
            - Calculating gradients is similar finding a steepest downward slope, we use the **magnitude of the gradient** (steepness of a slope) to tell us how big a step to take.
            - To decide on the step size, we multiply the gradient by a number we choose called the **learning rate.**
            - We then iterate until we have reached the lower point, and then stop.

            # The MNIST Loss Function

            `x` —> Independent variable 

            `y` —> dependent variables

            Now concatenating all the images (x) from a list of matrices (rank - 3) to a list of vectors (rank - 2). 

            Doing this by using Pytorch `**view**` , which changes the shape of a tensor without changing its content

            ```jsx
            train_x = torch.cat([stacked_threes , stacked_sevens]).view(-1 , 28*28)

            train_x.shape

            ```

            ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-23_at_16.03.08.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-23_at_16.03.08.png)

            `-1` —> denotes the row, since we don't know how many rows exactly in a dataset (or) this image, we use -1. Which says make this axis as big as necessary to fit all the data. Like we do in slicing!

            For labels, will use

            - **1** for `3s`
            - **0** for `7s`

            ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-23_at_16.10.43.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-23_at_16.10.43.png)

            By using `unsqueeze` we are adding a new dimension that vector. 

            `tensor([1] * len(threes) + [0] * len(sevens)).shape` —> vector but by adding a `unsqueeze(1)` it converts into a matrix of `(x , 1)` 

            After getting out `train_x` and `train_y` the next thing to do is to put them in a **Dataset**.

            A Dataset in Pytorch is required to return a tuple of (x,y) when indexed.

            ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_02.10.15.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_02.10.15.png)

            - **Initialize the parameters**

                [Why Initialize a Neural Network with Random Weights? - Machine Learning Mastery](https://machinelearningmastery.com/why-initialize-a-neural-network-with-random-weights/)

                Now the next step (or) the first step would be **initialize the parameters.**(Initially random) weight for every pixels.

                ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_02.38.04.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_02.38.04.png)

            The function `**weight * pixels`** won't be just enough, because image the scenario when our `pixels` are **0.** When we **multiply** our weights with a **pixel of value 0,** then the weight turns out to be **0** as well. 

            To overcome this we use something called **bias** (b), looking at this formula `y = wx + b` w and x would be our weight and pixels when those turns out to be **zero.** The bias (b) would be come in rescue. 

            `bias = init_params(1)` —> we initialize it constants.

            Together the `weights` and `biases` make up the parameters.

        - **Calculate the predictions**

            The next step would be calculate the gradients with the parameters we derived. 

            But let's first calculate the predictions manually before getting the help of **gradients**

            Likewise, we gotta calculate this for every row in the matrix, which is `w * x`. For loop would be slow, we use **matrix multiplication** here to speed up the process.

            In Python, matrix multiplication is represented with a `@` operator.

            ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_02.58.22.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_02.58.22.png)

      Calculating accuracy ,

    ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_03.06.05.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_03.06.05.png)

    Tried manually tweaking one value of **weight** with a hope that it would make a difference in the predictions. But it doesn't 

    So we need gradients in order to improve our model using SGD, and to calculate gradients we need a ***loss function*** that represents how good our model is. 

    Gradients are a measure of how that loss function changes with small tweaks to the weights.

- **Calculating the loss**

    We gotta choose a loss function, which would calculate our predictions for each image, collect these values to calculate an overall accuracy and then calculate the gradients of each weight with respect to that overall accuracy. 

    Since gradient of a function is its slope, that is how much the value of the function (slope) goes up or down, divided by how much we changed the input. 

    Mean the change in gradients depends on the input we divide by. This brings us a technical problem here but before that we can write mathematically the calculation of gradient, 

    `(y_new - y_old) / (x_new - x_old)`

    At times when  `x_new` is very similar to `x_old` then their differences will be very small. The change in accuracy happens only when the prediction changes from 3 to a 7, vice versa. 

    The problem is that a small change in weights from `x_old` to `x_new` is not likely to cause any change in the prediction. With that being said, `(y_new - y_old)` will almost always will be **0.** 

    For instance, (1 - 1) —> will be 0. 

    A very small change in the value of a weight will often not change the accuracy.

    Which means it's not useful to use **accuracy** as a loss function, if we then most of our time our gradient will be **0** and model will not be able to learn well.

    We need a loss function that when our weights result in slightly better predictions, gives us a slightly better loss. 

    For instance, if the correct answer is a 3, the score is little higher or if the correct answer is a 7, the score is little lower.

    The loss function receives not the image but the predictions `prds` from the model whether its a 3 or not (0 or 1). Then another argument `trgts` with the value 0 to 1.

    ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_05.04.01.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_05.04.01.png)

    `prds` —> 0.9 denotes it predicted the target above with 0.9 confidence that it's an 3. 

    **Decoding `torch.where` ,**

    This is same as list comprehension, except works on tensors, at C/CUDA speed. 

    This function will measure how distant each predictions is from 1 if it should be 1, and how distant it is from 0 if it should be 0, and then it will take the mean of all those distance.

    Since we need a scalar we take a mean, below list are predictions.

    `torch.where(trgts == 1 , 1- prds , prds)` —>

    `tensor([0.1000, 0.4000, 0.8000])`

    ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_05.14.34.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_05.14.34.png)

    As given above, since the predictions are quite accurate than before our loss will go down. 

    `mnist_distance` has a problem that it assumes that predictions are always between 0 and 1. We need a function which could put different values between 0 and 1.

    ### Sigmoid Function

    The sigmoid function always outputs a number between 0 and 1.

    `def sigmoid(x): return 1 / (1 + torch.exp(-x) )`

    But in Pytorch, we have a function which could do this for us. And this is an important function in deep learning, since we often want to ensure that values between 0 and 1.

    Below is the sigmoid generated using Pytorch

    ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_05.54.45.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_05.54.45.png)

    **We can see that it takes any input value, positive or negative, smooshes it into an output value between 0 and 1.** 

    Now our loss function will work even if the predictions are not between 0 and 1.

    - `metric` --> drive human understanding (for humans)
    - `loss` --> drive automated learning (for automatic updation)

    The loss function is calculated for **each item in our dataset** ,  and then at the end of an epoch, the **loss values are all averaged and the overall mean is reported for the epoch.**

    **Metrics** on other hand are the numbers we care about and these are the values which are printed at end of each epoch that tells us how our model is doing.

    When judging a performance of a model it's important to focus on metrics rather than the loss.

    ## SGD and Mini - Batches

    As we recreating those seven steps one by one for our Mnist data, now we have our loss function which is suitable for driving SGD. 

    The next phase would be update the weights based on the gradients. This is called an **optimization step.**

    To take an **optimization step** we need to calculate loss over one or more data items. The two ways we could calculate are: 

    - Calculate for the whole dataset.
    - Calculate for single data item (one by one).

    Either of those above options won't be ideal, that would produce an unstable gradient and it will cause a trouble on updating the gradients. After all this would be slow, imagine doing this for dataset which has million's of images. 

    The fix, 

    Instead we calculate the average loss for a few data items at a time. This is called **mini - batch**  and the number of data in mini batch is called the **batch size.** 

    Larger batch size —> more accurate and stable estimate of the dataset's gradient and loss function. But takes time since we will process fewer mini-batches per epoch. 

    Choosing a good batch size is one of the decisions one need to make a deep learning practitioner to train the model quickly and accurately. 

    Another good reason for using mini-batches, these **calculations** of batches takes place on **GPU**. These accelerator (GPU) perform well only if they have lots of work to do at a time, so it's helpful if we can give them lots of data. 

    A effective way is, putting our data on random shuffle on every epoch, before we create mini-batches. Below article **will explain the need for shuffling before creating mini batch.** 

    [Why should we shuffle data while training a neural network?](https://stats.stackexchange.com/questions/245502/why-should-we-shuffle-data-while-training-a-neural-network)

    The reasons would be: 

    - it prevents any bias during the training
    - it prevents the model from learning the order of the training
    - it helps the training converge fast

    Pytorch and Fastai provides `class` that will do the shuffling and mini-batch collation, called **`DataLoader`**

    A DataLoader can take any Python collection and turn it into iterator over many batches.

    ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_06.38.43.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_06.38.43.png)

    But for training a model, we don't just want any Python collection, but a collection containing **Independent** and **dependent** variables (inputs and targets of model). 

    A collection that contains tuples of Independent and dependent variables is known in Pytorch as a `Dataset` .

    ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_08.37.32.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_08.37.32.png)

    When we pass a `Dataset` to a `DataLoader` we will get back many batches that are themselves tuples of tensors representing batches of independent and dependent variables. 

    Instead of `coll` in previous example we could pass `ds` which is a dataset to a `DataLoader`

    ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_08.43.46.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_08.43.46.png)

    As we observe those tuples, the `DataLoader` structure those in such a way that both `x` and `y` values will be equally parted. The rest remaining tuples for example above, 

    `(tensor([25, 21]), ('z', 'v'))]` 

    those are tuples which couldn't form a (**6 , 6)** pair with other tuples. These pair's later equally distributed. 

    # Putting it All Together

    Now we experimented all those seven steps one by one, its time to implement as per the figure.

    ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Untitled.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Untitled.png)

    Our code will be implemented as below

    ```jsx
    for x , y in dl:
      pred = model(x)
      loss = loss_func(pred , y)
      loss.backward()
      parameters -= parameters.grad * lr
    ```

    ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_09.16.13.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_09.16.13.png)

    ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_09.16.32.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_09.16.32.png)

    ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_09.16.48.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_09.16.48.png)

    When we run the function `calc_grad` twice there was a change in the gradients.

    The reason was `loss.backward` adds the gradients of loss to any gradients that are currently stored.

    So we gotta make it zero.

    ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_09.17.09.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_09.17.09.png)

    After all these steps there exits one remaining step, it is to **update** the **weights** and **biases** based on the gradient and learning rate. 

    Likewise we made the **grad** zero up there for the parameters we will be doing the same thing right here too to avoid the confusion when we try to compute the derivative to the next batch. 

    Even here by using `grad.data` we are telling Pytorch not to take the gradient of the current step.

    Now its the time to construct a function which would calculate the accuracy, to decide if an output represent a 3 or a 7, we can check whether its greater than 0. 

    If it's greater than 3 if not 7, following this will take a mean of all the `True` so we will get an accuracy.

    ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_10.38.23.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_10.38.23.png)

    ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_10.38.44.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_10.38.44.png)

    ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_10.39.00.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_10.39.00.png)

    The functions, 

    - `train_epoch` —> does the training step, in short the seven steps we had seen before from intializing to gradients and updating.
    - `validate_epoch` —> calculates the accuracy for the trained model, from the train epoch.

    ## Creating an Optimizer

    We are close to the accuracy of the **pixel similarity approach.**

    The next step would be create an object that will handle the SGD step for us, in Pytorch its called an optimizer. In this step we will simplify our code by using Pytorch classes and make it easier to implement.

    Firstly, will repalce our linear function `linear1` with Pytorch's `nn.Linear` class. A module is an object of a class that inherits from the Pytorch `nn.Module`

    `nn.Linear` --> does same thing as `init_params` and `linear1` together. And it even takes care of weights and biases in a single class.

    ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_13.00.55.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_13.00.55.png)

    Below we are creating our own optimizer putting them into a class. 

    ![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_13.00.18.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_13.00.18.png)

    [Why do we need to use `zero_grad()` ?](https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch#:~:text=In%20PyTorch%20%2C%20we%20need%20to,the%20gradients%20on%20every%20loss.) 

Instead of all the mess, by creating a class and instantiating it. We can now use `opt` and call a method in place of the updation and zero grad.

```
def train_epoch(model , lr , params):
  for xb , yb in dl: # training loop for one epoch
    calc_grad(xb , yb , model)
    for p in params:
      p.data -= p.grad*lr # Updation 
      p.grad.zero_()
 
```

We modified the above code by creating a class and making some methods to simplify the steps. Into something like this

```
def train_epoch(model):
  for xb, yb in dl:
    calc_grad(xb , yb , model)
    opt.step()
    opt.zero_grad()
```

In place of `p` , `data` etc.. we replaced with the method we created in the class `BasicOptim()`

So now, we are putting our training loop in the function

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_13.13.23.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_13.13.23.png)

The good thing is **fastai** provides the `SGD` class, that by default does the same thing as our `BasicOptim()` class. 

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_13.16.26.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_13.16.26.png)

Rather using the `train_model` fastai also provides `Learner.fit` which we can use instead of the `train_model.`

But before to create a `Learner` we first need to create a `DataLoaders` (note the s), by passing in our training and validation `DataLoader` which is ( `dl` and `valid_dl` ) 

```python
# Creating the DataLoaders, from the two DataLoader --> to use Learner

dls = DataLoaders(dl , valid_dl)
```

To create a `Learner` without using the application (`cnn_learner`) we need to pass in all the elements we created so far, 

- DataLoaders
- the model
- the optimization function
- the loss function
- other metrics we created

Using `Learner` from fastai to fit the model, 

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_14.04.02.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-25_at_14.04.02.png)

## Adding a Nonlinearity

So far we been dealing with simple linear classifier, a linear classifier is constrained with what it can do, but to make it more bit complex and able to handle more tasks. 

We need to add something nonlinear between two linear classifiers which gives us a neural network.

```python
# Constructing a basic neural network 

def simple_net(xb):
  res = xb@w1 + b1
  res = res.max(tensor(0.0)) # activation function
  res = res@w2 + b2
  return res
```

The above is the basic neural network, with two linear classifiers with a max function between them 

`w1` & `w2` —> are weight tensors 

`b1` & `b2` —> bias tensors 

Even here the parameters `w` and `b` are initially randomly initialized. 

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-26_at_17.32.00.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-26_at_17.32.00.png)

In the above picture, `w1` has 30 output activations which means `w2` must have 30 input activations, so they match.

Well that means the first layer can construct 30 different features, each representing a different mix of pixels. 

The function `res.max(tensor(0.0))` is called as *rectified linear unit*, also known as **ReLU.** All it does is replace negative number with a zero. We can use it by `F.relu` in Pytorch. 

**Why there is a need for non-linear layers?**

[Why Are Neural Nets Non-linear?](https://medium.com/swlh/why-are-neural-nets-non-linear-a46756c2d67f)

By using more linear layers we can have our model do more computations, but there is no point of stacking up tons of linear layers directly after one another. 

Because rather adding and multiplying them number of times then it would be efficient if something replaced by multiplying different things together and adding them up just once. 

**In short, a series of linear layers can be replaced by a single layer with different set of parameters.**

*The non-linear layers enable neural nets to learn making conditional decisions for controlling the computational flow.*

When we put a `non - linear` layer between every linear layer, now each linear layer is somewhat decoupled from the other ones and now can be more efficient in doing its work. 

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-26_at_18.11.33.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-26_at_18.11.33.png)

Cutting down our `simple_net` function in more shorter lines and ease up the process of writing code for the neural net. 

The three lines of code that we have here are known as **layers.** The first and third layer known as **linear layers** and the 2nd layer known as *nonlinearity* (or) **activation function**

`nn.Sequential` —> creates a module that will call each of the listed layers (3) in turn. 

`nn.ReLU` == `F.relu`

Since `nn.Sequential` is a module, we can peek in the parameters of all modules it contains. 

And its a deeper model than we trained on before, so we will lower the learning rate and use few more epochs to balance.

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-26_at_18.21.17.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-26_at_18.21.17.png)

The whole training process is recorded in `learn.recorder` with the table of output stored in the **value** attribute. 

![Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-26_at_18.26.40.png](Under%20the%20hood%20Training%20a%20Digit%20Classifier%201e2f1ff0fcdf436fa55b1c0b6e13373b/Screenshot_2021-01-26_at_18.26.40.png)

Well now we have two things, 

- A function that can solve any problem to any level of accuracy by giving the correct set of parameters (neural net).
- A way to find the best set of parameters for any function (SGD).

**About Deeper model**

For deeper model we don't need to use **many parameters**, it turns out that we can use **smaller matrices**, with more layers and get better results than we would get with larger matrices with just few layers.

Neural network contains a lot of numbers, but they are only two types of numbers that are calculated and the parameters that these numbers are calculated from. 

***Activations***

Numbers that are calculated both by linear and non - linearity. 

***Parameters***

Numbers that are randomly initialized and optimized ( numbers that define the model). 

**Part of becoming a good deep learning practitioner is getting used to the idea of looking at your activations and parameters, and plotting them and testing whether they are behaving correctly.**