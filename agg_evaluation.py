import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

covariates = ["attendance_1", "rating_1", "talktime_tutor_pct_1",
    "spoken_token_tutor_pct_1", "chat_token_tutor_pct_1", "length_utterance_tutor_1",
    "length_utterance_student_1", "length_utterance_tutor_chat_1", "length_utterance_student_chat_1",
    "ratio_students_engaged_1", "normalized_num_turns_1",
    "normalized_num_high_uptakes_1", "normalized_num_eliciting_1", "normalized_num_questions_students_1",
    "normalized_num_questions_tutor_1", "normalized_student_reasoning_1", "min_sat_score_series",
    "max_sat_score_series", "grade_for_session_1"]

outcome_vars = ['grade_change', 'uptake_change', 'eliciting_change', 'talktime_change']

arms = ['control', 'tutor', 'tutor_student_personal', 'tutor_student_social']

outcomes = ['uptake', 'eliciting', 'grade', 'talktime']

df = pd.read_csv('agg_data_with_policy_ridge.csv')

def get_avg_scores(df, policy_col, outcome_var_col):
    matching_df = df[df['arm_name'] == df[policy_col]]
    aligned_mean = matching_df[outcome_var_col].mean()

    not_matching_df = df[df['arm_name'] != df[policy_col]]
    not_aligned_mean = not_matching_df[outcome_var_col].mean()

    return aligned_mean, not_aligned_mean

# get policy performance for all outcomes
# TODO: need to run for both ridge and lasso, decide on how we want to save scores

uptake_al_score, uptake_not_al_score = get_avg_scores(df, 'uptake_policy', 'uptake_change')
print("Uptake aligned score: ", uptake_al_score)
print("Uptake not aligned score: ", uptake_not_al_score)

eliciting_al_score, eliciting_not_al_score = get_avg_scores(df, 'eliciting_policy', 'eliciting_change')
print("Eliciting aligned score: ", eliciting_al_score)
print("Eliciting not aligned score: ", eliciting_not_al_score)

grade_al_score, grade_not_al_score = get_avg_scores(df, 'grade_policy', 'grade_change')
print("Grade aligned score: ", grade_al_score)
print("Grade not aligned score: ", grade_not_al_score)

talktime_al_score, talktime_not_al_score = get_avg_scores(df, 'talktime_policy', 'talktime_change')
print("Talktime aligned score: ", talktime_al_score)
print("Talktime not aligned score: ", talktime_not_al_score)