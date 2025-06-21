library(tidyverse)
library(arrow)

tmp <- read_parquet("c:\\users\\fharren\\downloads\\heart_big_train.parquet")
names(tmp)

sampled_data <- tmp |> slice_sample(n = 1000, replace = FALSE)

write.table(sampled_data,
            file = "c:\\users\\fharren\\downloads\\heart_big_train_sample.csv",
            sep = ",",
            dec = ".",
            row.names = FALSE,
            quote = TRUE,
            qmethod = "double"
)

