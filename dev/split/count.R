library(tidyverse)
library(arrow)

tmp <- read_parquet("C:\\Users\\fharren\\Desktop\\projecten HU\\05 Machine Learning\\hufrans\\mads-exam-fharren\\data\\heart_big_train.parq")
tmp |> count(target)
