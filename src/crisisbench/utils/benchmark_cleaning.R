library(tidyverse)

gsheet <- "https://docs.google.com/spreadsheets/d/1BLOp3QOoAKWUEWkF8n4yrYAQk1vqTE3GVdjGAD6LneY/edit?gid=1831781067#gid=1831781067"

sample_cases_sheets <- googlesheets4::sheet_names(gsheet)

sample_cases <- map_dfr(sample_cases_sheets,
                    ~googlesheets4::read_sheet(ss = gsheet, sheet = .x) %>% 
                      mutate(case = .x))

sample_case_df_clean <- sample_cases %>% 
  transmute(id = paste0(case, "_", `Case ID`),
            sender = Sender,
            message = Message) %>% 
  group_by(id) %>% 
  mutate(order = row_number()) %>% 
  ungroup() %>% 
  filter(!is.na(id),
         !is.na(message)) %>% 
  mutate(sender = case_when(grepl("Client|client", sender) ~ "client",
                            grepl("Counsellor|Counselor|counselor|counsellor", sender) ~ "counselor"),
         sender = coalesce(sender, lag(sender))) %>% 
  select(id, order, sender, message) %>% 
  group_by(id) %>% 
  mutate(to_combine = coalesce(sender == lag(sender), F),
         sender_id = cumsum(as.numeric(!to_combine))) %>% 
  ungroup() %>% 
  group_by(id, sender_id, sender) %>% 
  summarize(message = paste(message, collapse = " ")) %>% 
  ungroup() %>% 
  group_by(id) %>% 
  mutate(order = row_number()) %>% 
  select(id, order, sender, message) %>% 
  filter(!(row_number() == n() & sender == "counselor"))

googlesheets4::write_sheet(ss = gsheet,
                           data = sample_case_df_clean,
                           sheet = "combined")

benchmark <- readr::read_csv("~/vf_copilot/evals/data/benchmark/benchmark_processed2.csv")
results1 <- readr::read_csv("/Users/akshayswaminathan/vf_copilot/prompt_eng/outputs/prompt_experiment_outputs_2024-08-18_00:40:49.csv")


get_last_message_text <- function(message1, message2) {
  
  str_replace(str_replace_all(message2, "\n", " "), str_replace_all(message1, "\n", " "), "") %>% 
    str_replace_all("\n", " ") %>% 
    str_extract("counselor:.*client:") %>% 
    str_replace("client:.*", "") %>% 
    str_replace("counselor: ", "") %>% 
    trimws() %>% 
    # str_replace("\nclient:.*\n$", "") %>% 
    # str_replace("^counselor: ", "") %>% 
    return()
  
}


sample_bench <- benchmark %>% 
  group_by(id) %>% 
  mutate(split_messages = str_split(conversation_history, "client:|counselor: "),
         diff = map2_chr(split_messages, lead(split_messages), ~setdiff(.y, .x)[1] %>% paste(collapse = " "))) %>% 
  transmute(id, conversation_history, human = diff) 

sample_bench$conversation_history[19]

filled_bench <- readr::read_csv("/Users/akshayswaminathan/vf_copilot/prompt_eng/outputs/prompt_experiment_outputs_2024-08-18_00:40:49.csv")

final_filled_bench <- filled_bench %>% 
  left_join(sample_bench) %>% 
  filter(human != "") 

final_filled_bench %>%   
  write_csv("~/vf_copilot/prompt_eng/outputs/prompt_experiment_outputs_2024-08-18_with_human_responses.csv")


scores <- readr::read_csv("/Users/akshayswaminathan/vf_copilot/evals/data/benchmark/benchmark_scores_2024-08-06_22-35-33.csv")
cleaned_scores <- scores %>% 
  transmute(
    exp_id,
    model = model_output,
    guideline = normalized_score,
    forbidden = Total_Forbidden_Statements,
    n_tasks = `Total Number of Tasks`,
    si_present = `Suicidal Thinking Present`,
    risk_assessment_conducted = `Risk Assessment Conducted`,
    risk_assessment_score = case_when(si_present == 0 & risk_assessment_conducted == 0 ~ 1,
                                      si_present == 0 & risk_assessment_conducted == 1 ~ 1,
                                      si_present == 1 & risk_assessment_conducted == 0 ~ 0,
                                      si_present == 1 & risk_assessment_conducted == 1 ~ 1)) %>% 
  group_by(model, exp_id) %>% 
  nest() %>% 
  mutate(cleaned = map(data, function(df) map_df(df, ~Filter(function(x) !is.na(x), .x)))) %>% 
  select(-data) %>% 
  unnest(cleaned)

cleaned_scores %>% 
  write_csv(glue::glue("~/vf_copilot/evals/benchmark_scores_cleaned_{Sys.Date()}.csv"))


  
cleaned_scores %>% 
  filter(model == "human") %>% 
  filter(guideline > 0.4) %>% 
  left_join(final_filled_bench %>% 
              select(exp_id, conversation_history, human), "exp_id") %>% 
  left_join(scores %>% 
              select(model = model_output,
                     rationale,
                     exp_id) %>% 
              filter(!is.na(rationale))) %>% 
  select(exp_id, guideline, conversation_history, human, rationale) %>% 
  googlesheets4::write_sheet(ss = "https://docs.google.com/spreadsheets/d/1P6qO9V__vTZlUjACwq0H6h857bbn8JhBNZi3c6IILTc/edit?gid=0#gid=0",
                             sheet = as.character(Sys.Date()))




