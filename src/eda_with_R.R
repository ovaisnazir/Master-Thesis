

setwd("D:\\Master\\Asignaturas\\TFM\\Git\\TFM\\data")

# Load libraries (and install them when needed)
packages<- c("data.table","ggplot2","tidyverse","tidyr","dplyr"
             ,"tibble","forecast","tsfknn","anytime","varhandle","lubridate","nortest"
             ,"normtest","scales","xts","tsfknn","TSrepr","DataExplorer","ExPanDaR")

loadLibraries <- function(pakages) {
  usePackage <- function(p){
    if ( !is.element(p, installed.packages()[, 1]) ) {
      install.packages(p, dep = TRUE)}
    require(p, character.only = TRUE)}
  
  for (p in packages){ usePackage(p) }
  
} 

loadLibraries(packages)

## Loading data sets

# raw data
X_train <- as.data.frame(read.csv(file = "./raw/X_train_v2.csv", sep = ","))
y_train <- as.data.frame(read.csv(file = "./raw/Y_train.csv", sep="," ))
X_test <- as.data.frame(read.csv(file = "./raw/X_test_v2.csv", sep = ","))

# Complementary data
extra_data <- as.data.frame(read.csv(file = "./external/WindFarms_complementary_data.csv", sep = ";"))

# EDA transformed data for WF1
# train_WF1 <- as.data.frame(read.csv(file = "./interim/by_WF/train/df_WF1.csv", sep = "," ))
# test_WF1 <- as.data.frame(read.csv(file = "./interim/by_WF/test/df_WF1.csv", sep = "," ))

X_train$Time <- unfactor(X_train$Time)
X_train$Time <- as.POSIXct(X_train$Time, format = "%d/%m/%Y %H:%M", tz="GMT")

X_test$Time <- unfactor(X_test$Time)
X_test$Time <- as.POSIXct(X_test$Time, format = "%d/%m/%Y %H:%M", tz="GMT")


df <- merge(X_train,y_train,by="ID")
df <- select(df, c('ID','WF','Time','NWP1_00h_D_U','NWP1_00h_D_V',
                   'NWP1_00h_D_T','NWP2_00h_D_U','NWP2_00h_D_V','NWP3_00h_D_U',
                   'NWP3_00h_D_V','NWP3_00h_D_T','NWP4_00h_D_U','NWP4_00h_D_V','NWP4_00h_D_CLCT','Production'))

create_report(df, y = "Production")
plot_qq(df, by = "WF")

install.packages("ExPanDaR")
library("ExPanDaR")
ExPanD(df = df, cs_id = "ID", ts_id = "Time")