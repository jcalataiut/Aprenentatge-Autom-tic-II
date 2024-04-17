library(torch)

# Create torch from atomic vector
x<- torch_tensor(c(1,2,3)); x

# Create torch from matrix
m<-matrix(runif(6), nrow=2)
x<-torch_tensor(m)
x

# Create torch from array (R per defecte indexa a la ultima i PyTorch a la primera)
a<-array(runif(16),dim=c(4,2,2))
x <- torch_tensor(a)
x

cuda_is_available()

## Using initilization functions
# Return a tensor filled with values drawn rom a unit normal distribution
x<-torch_randn(c(5,2,3))
x

torch_zeros(5)

## Converting back to R
# torch provide methods e.g as.matrix, as.array, as.numeric, as.integer to convert tensor back to R

x<- torch_randn(2,2)
as.array(x)


##Tensor attributes
# 1. data type (tipus de dades)
# 2. device 
# 3. dimensions (dimensions)
# 4. require_grad 

x <- torch_randn(2,2); x

# Acess data type
x$dtype

# Acess devide
x$device

# Acess dimensions
x$shape

# Require gradient 
x$requires_grad


### Modifying attributes
## Default tensor attributes can be modified whan creating the tensors or later using te $to method

# Modify tensor during creation
x<- torch_randn(2,2, dtype = torch_float64())
x

# Modify using $to
x <- torch_randn(2,2)
x$dtype
x <- x$to(dtype=torch_float16())
x$dtype


### CUDA devices
## Moving between devices is also done with the $to method, but only cpu devices
## are available to all systems. Moving between devices is an important operation because tensor
## operations happen on the device where the tensor is located; so if you want to use the fast GPU implementations, 
## you need to move tensors to he CUDA device. A common pattern in torch is to create a device object at the 
## beginning of your script and reuse it as you create and move tensord. For example:

# Create device:
device <- if(cuda_is_available()) "cuda" else "cpu"
device <- ifelse(cuda_is_available(), "cuda", "cpu")

x <- torch_randn(2,2, device = device)
x


y <- x$to(device=device)
y


### Indexing tensors
## Indexing tensors in torch is very similar to indexing vectors, matrices and arrays in R
## (with an important difference when using negative indexes)

## In torch negative indexes don't remove the element, instead selection happens starting form the end which 
## is used more frequently

x<- torch_tensor(1:5); x

# Take firts element
x[1]

# Negative index from last
x[-1]

# Select firts 3 elements
x[1:3]

# Selecting from 3rd element to last using N
x[3:N]


# Select the last 2 elements
x[-2:N]

# Select using a boolean tensor
x[x>2]


### Multidimensional selections
## When indexing a tensor with multiple dimensions, you can use dimension-specific
## indices separated by commas, just like in R. For example:

x <- torch_randn(2,2,3) ;x

# Selecting firts element in every dimension
x[1,1,]

# Select everything from a dimension using empty argument (pilla de cadascun la primera columna i ho disposa per files)
x[,,1]

x[,1,] # el mateix per per la primera fila de cadascun

# de forma anàloga es pot fer
x[..,1]


# You can also add a new dimension using the newaxis sugar
x[.., newaxis] # pila totes les possibles combinacions de les diferents dimensions que deixa lliure


# By default when you select a single element from a dimesions it's droppen
# you can change this behavior by setting drop=FALSE

x[1,..]

# Subset assignment is also supported
x[1,1,1] <- 0
x[1,1,1]


### Array Computation
## torch provides more than 200 funtions and methos that operate on tensors.
## They range from mathematical operations to utilities for reshaping and modifying tensors.

## Most operations have both CPU and GPU backends, and torch will use the backend corresponding to
## the tensor device.

x <- c(1,2,3) %>% torch_tensor(); x

# Subtract other scaled by alpha
x %>% torch_sub(1) # resta 1 a cada component 

# many torch_* functions have a corresponding tensor method
x$sub(1) # fa el mateix qu la anterior però simplificada

x %>% 
  torch_exp() %>% 
  torch_log()


## Reduction functions

x <- rbind(c(1,2,3), 4:6) %>% 
  torch_tensor(); x


# Sum of all elements in the input tensor
x$sum()


# Reduce the firts dimension, ie, sum all rows for each column
# Reduce rows by adding columns?

x %>% 
  torch_sum(dim=1) # en la primera dimensió reduce per rows

# Reduce the 2nd dimension: sum all columns for each row
# Reduce columns by adding rows?

x %>% 
  torch_sum(dim=2) # en la primera dimensió reduce per columns

## Broadcasting 
# Allows one to use tensors of different shapes when executing binary/arithmetic operations

# Simple broadcasting example 
torch_tensor(c(1,2,3)) + 1

# Adding a (3,2) matrix to a (2) vector
torch_ones(3, 2) + torch_tensor(c(1, 2)) # suma per files

torch_ones(2,3) + torch_tensor(c(1,2,3))


# Danger will robinson
torch_ones(10, 1) + torch_tensor(rep(1, 10))



