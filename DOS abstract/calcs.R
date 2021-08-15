
pacman::p_load(tidyverse, tidymodels, here)

# function to calculate sens and spec

calc_sns_spc <- function(col, type) {
  tp <- sum(truth[[col]] == 1 & preds[[col]] == 1)
  tn <- sum(truth[[col]] == 0 & preds[[col]] == 0)
  fp <- sum(truth[[col]] == 0 & preds[[col]] == 1)
  fn <- sum(truth[[col]] == 1 & preds[[col]] == 0)
  if (type == "sens") {
    out <- Hmisc::binconf(tp, tp + fn, method = "wilson")
  }
  if (type == "spec") {
    out <- Hmisc::binconf(tn, tn + fp, method = "wilson")
  }
  return(out)
}


cols <- c("abn", "acl", "men")

probas <- read.csv(here("submit_test/all_valids_preds.csv"),
  header = F,
  col.names = cols
)

preds <- probas %>%
  mutate(
    across(.fns = ~ if_else(.x >= 0.5, 1, 0)),
    across(.fns = as.factor)
  )


truth <- list.files(here("data"), pattern = "valid-.") %>%
  map2_dfc(
    .x = .,
    .y = cols,
    .f = ~ read.csv(here("data", .x), header = F, col.names = c("id", .y))
  ) %>%
  select(all_of(cols)) %>%
  mutate(across(.fns = as.factor))

# dfs <- cols %>% map(
#  ~tibble(truth[.x], probas[.x], preds[.x], .name_repair = ~set_names(c("truth", "probas", "preds")))) %>%   set_names(cols)

aucs <- cols %>%
  map(~ pROC::roc(response = truth[[.x]], predictor = as.numeric(probas[[.x]]))) %>%
  map(~ c(pROC::ci.auc(.x, method = "delong"))) %>%
  set_names(cols)

sens <- cols %>%
  map(~ calc_sns_spc(.x, "sens")) %>%
  set_names(cols)

spec <- cols %>%
  map(~ calc_sns_spc(.x, "spec")) %>%
  set_names(cols)
