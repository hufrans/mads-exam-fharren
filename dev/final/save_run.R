library(tidyverse)
library(arrow)
library(openxlsx)
library(janitor)


path <- file.path("C:/Users/fharren/Desktop/projecten HU/05 Machine Learning/hufrans/mads-exam-fharren")
input_path <- file.path(path,"runs")
output_path <- file.path(path,"output")
data_path <- file.path(path,"data")


run_basic <- read_parquet(file.path(input_path,"20250623-183016_all_model_summary_final.parquet")) |> mutate(run = "basic") |> clean_names()
run_tuned <- read_parquet(file.path(input_path,"20250624-172550_all_model_summary_final.parquet")) |> mutate(run = "tuned") |> clean_names()

data_basic <- read_parquet(file.path(data_path,"heart_big_train.parq")) |> mutate(run = "basic") |> clean_names()
data_tuned <- read_parquet(file.path(data_path,"heart_big_train_synthetic.parquet")) |> mutate(run = "tuned") |> clean_names()
data_test <- read_parquet(file.path(data_path,"heart_big_test.parq")) |> mutate(run = "test") |> clean_names()


run <- bind_rows(run_basic, run_tuned)

run |>
  write_parquet(file.path(output_path,"stats.parquet"))
run |>
  unnest_wider(model_specific_config) |>
  write_csv(file.path(output_path,"stats.csv"))
run |>
  unnest_wider(model_specific_config) |>
  write.xlsx(file.path(output_path,"stats.xlsx"))






data <- bind_rows(data_basic, data_tuned, data_test)

counts_df <- data |> count(run, target) |> pivot_wider(names_from = c(target), values_from = n)

percentages_df <- counts_df %>%
  adorn_percentages("row") %>% # Bereken percentages per rij
  adorn_pct_formatting(digits = 2) %>% # Formatteer als percentage met 2 decimalen
  # Hernoem de kolommen om duidelijk te maken dat dit percentages zijn
  rename_with(~ paste0("perc_", .x), .cols = -run)

combined_table <- inner_join(counts_df, percentages_df, by = "run") |>
  select(run,`0`,perc_0,`1`,perc_1,`2`,perc_2,`3`,perc_3,`4`,perc_4)


print(combined_table)

data |> count(run)
