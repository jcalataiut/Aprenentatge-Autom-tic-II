pacman::p_load(tensorflow)
pacman::p_load(keras3)
keras3::install_keras(backend="tensorflow")


x <- 1:15
prod_ac <- op_cumprod(x) 

#internament implementa la estructura i defineix el graf computacional, i açò és més
# eficient ja que optimitza el càlcul i permiteix fer operaciones en paralel i reduir el temps d'execució 

eval(prod_ac)

sig <- op_sigmoid(c(-3,-2,-1,0,1,2,3))
eval(sig)

eval(random_normal(c(4,6), mean=0, stddev = 1))

##############
# Exercici 1 #: Strain prediction
##############

library(foreign)
library(keras3)
library(caret)
library(tidyverse)
read.spss("files/BD_strain.sav", use.value.labels = TRUE, max.value.labels = Inf, to.data.frame = TRUE) %>%
  select(LASRs, LASs, RECURRENCIA) -> strain

strain[,c(1,2)]<- scale(strain[,c(1,2)])
head(strain)

lm <- lm(LASs~LASRs, data=strain)
summary(lm)

plot(strain$LASRs, strain$LASs, pch=16, cex=1.5)
abline(lm, type='l', lwd=3, col='red')


# una regressió lineal es una XN més sencilla, un sol input i sol output amb només una neurona
# un pes i un biax amb funció d'activació la identitat i funció de cost la LSE

# una XN és una seqüència de capes

linear_regression <- keras_model_sequential() #instanciem el model, definim el model
linear_regression %>% layer_dense(input_shape=1, units = 1, activation="linear")  # layer dense es el mateix que fer una capa fully connected neuronal

linear_regression %>% compile(loss="mse", optimizer = optimizer_sgd(learning_rate=0.001))

history <- linear_regression %>% fit(x=strain$LASRs, y=strain$LASs, epochs = 500) #per defecte fa una inicialitació aleatoria assumit una distribució normal en cada capa amb variància cte.

get_weights(linear_regression)[[1]]
get_weights(linear_regression)[[2]]

scores <- linear_regression %>% evaluate(strain$LASRs, strain$LASs) #calcula el error final 
print(scores)

y_pred <- linear_regression %>% predict(strain$LASRs)


fig_strain <- ggplot(strain, aes(LASRs, LASs)) + 
  geom_point(aes(colour=factor(RECURRENCIA)), size=4)+
  theme_grey(base_size=20) + 
  theme(legend.title = )


logistic_reg <- keras_model_sequential()
logistic_reg %>% layer_dense(input_shape=2, units = 1, activation = "sigmoid")

logistic_reg %>% compile(loss="binary_crossentropy", optimizer =  "adam")

history <- logistic_reg %>% fit(x=strain[-3] %>% as.matrix, y=as.numeric(strain$RECURRENCIA)-1, epochs = 100)

get_weights(logistic_reg)[[1]]
get_weights(logistic_reg)[[2]]

scores <- linear_regression %>% evaluate(strain$LASRs, strain$LASs) #calcula el error final 
print(scores)

data_pred_strain <- data.frame(expand.grid(LASRs=seq(-3,3,by=0.1),LASs=seq(-3,3,by=0.1)))
data_pred_strain$prediccio_keras <- logistic_reg %>% predict(as.matrix(data_pred_strain))

predict_lr <- ggplot(data_pred_strain) + 
  geom_tile(aes(x=LASRs, y=LASs, fill=prediccio_keras), show.legend=TRUE)+
  scale_fill_gradient(low = "blue", high = "red") + 
  ggtitle("LOGISTIC PREDICTION") + xlab("LASRs") +
  ylab("LASs") + theme_grey(base_size=15) + theme(legend.title=element_blank())


strain$prediccio_keras <- logistic_reg %>% predict(strain[-3] %>% as.matrix)

library(pROC)
corba_roc <- roc(strain$RECURRENCIA, strain$prediccio_keras %>% as.numeric)
auc(corba_roc); ci(corba_roc); plot(corba_roc)

roc_coords <- coords(corba_roc, "best", ret=c("threshold", "accuracy", "specificity", "sensitivity", "ppv", "npv"))
roc_coords


library(epiR)
threshold <- as.numeric(roc_coords[1])
conf_matrix <- table(strain$prediccio_keras>threshold, strain$RECURRENCIA)[c(2,1), c(2,1)]
epiR::epi.2by2(dat=conf_matrix, method="cohort.count", outcome="as.columns") %>% print


# PROBLEMA 4

EAM_model <- keras_model_sequential() %>% 
  layer_dense(input_shape=1, units = 64, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1, activation = "linear")

EAM_model %>% compile(loss="mse", optimizer = "adam")
EAM_model %>% summary()

x <- as.numeric(EAM_data$Realce_IIR)
y<- as.numeric(EAM_data$Voltaje)
EAM_model %>% fit(x,y,epochs = 100)
scores <- EAM_model %>% evaluate(x,y)
print(scores)

y_pred <- EAM_model %>% predict(x)
plot(x,y, pch=16)
points(x, y_pred, col="red", pch=16)

# Sembla que el model ajusta amb una relació negativa

## PROBLEMA 4

x <- seq(0,25, by=0.01)
y <- (1/(x+1))*(4*sin(x) + rnorm(n=length(x), 0, 1.5))
dades_sin <- data.frame(X=x, Y=y)
ggplot(dades_sin) + geom_point(aes(x=x, y=y), cex=2) + 
  theme_grey(base_size=20) + ylab('f(x)')

EAM_model <- keras_model_sequential() %>% 
  layer_dense(input_shape=1, units = 64, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1, activation = "linear")

EAM_model %>% compile(loss="mse", optimizer = "adam")
EAM_model %>% summary()

EAM_model %>% fit(x,y,epochs = 500)
scores <- EAM_model %>% evaluate(x,y)
print(scores)

y_pred <- EAM_model %>% predict(x)
plot(x,y, pch=16)
lines(x, y_pred, col="red", pch=16, lwd=2)

