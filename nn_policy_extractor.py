import torch
import torch.nn as nn
import pandas as pd

class PolicyPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_size, output_size)
        self.dropout2 = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        out = self.dropout1(x)
        out = self.lin1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.lin2(out)
        return out

def load_model(weights_path, input_size, hidden_size, output_size, dropout_prob):
    model = PolicyPredictor(input_size, hidden_size, output_size, dropout_prob)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model

def predict_and_add_column(df, preds_df, model, covariates, new_column_name='predictions'):
    if preds_df.empty:
            preds_df = pd.DataFrame(index=df.index)

    # Convert DataFrame to tensor
    X = torch.tensor(df[covariates].values).float()
    
    # Make predictions
    with torch.no_grad():
        predictions = model(X).numpy()
    
    # Add predictions to the DataFrame
    preds_df[new_column_name] = predictions
    return preds_df

def main():
    # Hyperparameters (must match those used during training)
    input_size = 19
    hidden_size = 10
    output_size = 1
    dropout_prob = 0.1
    weights_path = 'models/tutor_social_aggregate_change_model.pth'
    
    # Load the model
    model = load_model(weights_path, input_size, hidden_size, output_size, dropout_prob)
    
    all_cols = ["attendance_1", "rating_1", "talktime_tutor_pct_1",
    "spoken_token_tutor_pct_1", "chat_token_tutor_pct_1", "length_utterance_tutor_1",
    "length_utterance_student_1", "length_utterance_tutor_chat_1", "length_utterance_student_chat_1",
    "ratio_students_engaged_1", "normalized_num_turns_1",
    "normalized_num_high_uptakes_1", "normalized_num_eliciting_1", "normalized_num_questions_students_1",
    "normalized_num_questions_tutor_1", "normalized_student_reasoning_1", "min_sat_score_series",
    "max_sat_score_series", "grade_for_session_1", "arm_name", "eliciting_change", "talktime_change", "grade_change", "uptake_change"]

    covariates = ["attendance_1", "rating_1", "talktime_tutor_pct_1",
    "spoken_token_tutor_pct_1", "chat_token_tutor_pct_1", "length_utterance_tutor_1",
    "length_utterance_student_1", "length_utterance_tutor_chat_1", "length_utterance_student_chat_1",
    "ratio_students_engaged_1", "normalized_num_turns_1",
    "normalized_num_high_uptakes_1", "normalized_num_eliciting_1", "normalized_num_questions_students_1",
    "normalized_num_questions_tutor_1", "normalized_student_reasoning_1", "min_sat_score_series",
    "max_sat_score_series", "grade_for_session_1"]

    full_df = pd.read_csv("aggregated_data.csv")[covariates]
    full_df[covariates] = full_df[covariates].apply(pd.to_numeric, errors='coerce')

    preds_df = pd.read_csv("aggregated_data_w_preds.csv")
    # preds_df = pd.DataFrame()
    
    # Add predictions to the DataFrame
    preds_df = predict_and_add_column(full_df, preds_df, model, covariates, new_column_name='agg_3')

    preds_df.to_csv("aggregated_data_w_preds.csv", index=False)
    
    # print(df_with_predictions)

if __name__ == "__main__":
    main()