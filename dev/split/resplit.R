library(tidyverse)
library(arrow)

org <- read_parquet("C:\\Users\\fharren\\Desktop\\projecten HU\\05 Machine Learning\\hufrans\\mads-exam-fharren\\data\\heart_big_train.parq")
class_1 <- read_parquet("C:\\Users\\fharren\\Desktop\\projecten HU\\05 Machine Learning\\hufrans\\mads-exam-fharren\\gan_generated_output\\synthetic_ecgs_class_1_20250615_162144.parquet")
class_2 <- read_parquet("C:\\Users\\fharren\\Desktop\\projecten HU\\05 Machine Learning\\hufrans\\mads-exam-fharren\\gan_generated_output\\synthetic_ecgs_class_2_20250616_035905.parquet")
class_3 <- read_parquet("C:\\Users\\fharren\\Desktop\\projecten HU\\05 Machine Learning\\hufrans\\mads-exam-fharren\\gan_generated_output\\synthetic_ecgs_class_3_20250617_124628.parquet")
class_4 <- read_parquet("C:\\Users\\fharren\\Desktop\\projecten HU\\05 Machine Learning\\hufrans\\mads-exam-fharren\\gan_generated_output\\synthetic_ecgs_class_4_20250618_135429.parquet")

tot1 <- bind_rows(org, class_1, class_2, class_3, class_4)

tot2 <- tot1 |> slice_sample(prop = 1)

tot2 |> count(target)


write_parquet(tot2,
              "C:\\Users\\fharren\\Desktop\\projecten HU\\05 Machine Learning\\hufrans\\mads-exam-fharren\\data\\heart_big_train_synthetic.parquet",
              compression = "zstd", compression_level = 22)
org2 <- read_parquet("C:\\Users\\fharren\\Desktop\\projecten HU\\05 Machine Learning\\hufrans\\mads-exam-fharren\\data\\heart_big_train_synthetic.parquet")
