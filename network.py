import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

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

def load_data(arm, covariates, outcome_measure):
    train_data = pd.read_csv(f'full_data_splits/{arm}_test_data.csv')
    test_data = pd.read_csv(f'full_data_splits/{arm}_train_data.csv')

    train_data[covariates] = train_data[covariates].apply(pd.to_numeric, errors='coerce')
    test_data[covariates] = test_data[covariates].apply(pd.to_numeric, errors='coerce')

    train_data['aggregate_change'] = train_data['uptake_change'] + train_data['eliciting_change'] + train_data['talktime_change'] + train_data['grade_change']
    test_data['aggregate_change'] = test_data['uptake_change'] + test_data['eliciting_change'] + test_data['talktime_change'] + test_data['grade_change']
    
    X_train = torch.tensor(train_data[covariates].values).float()
    y_train = torch.tensor(train_data[outcome_measure].values).float()
    
    X_test = torch.tensor(test_data[covariates].values).float()
    y_test = torch.tensor(test_data[outcome_measure].values).float()
    
    return X_train, y_train, X_test, y_test

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=10):
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, loss_fn, optimizer, num_epochs, best_path):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            train_loss = loss_fn(outputs, y_batch)
            train_loss.backward()
            optimizer.step()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                outputs = model(x_batch)
                loss = loss_fn(outputs, y_batch)
                test_loss += loss.item()
        test_loss /= len(test_loader)

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), best_path)    

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}')
            print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.4f}')

    print("Training complete.")

def main():
    # 1. SELECT ARM AND OUTCOME
    # curr_arm = 'Control'
    # curr_outcome = 'uptake'

    arm_mapping = {'Control': 'ctrl', 'Tutor Feedback': 'tutor', 'Tutor Social': 'tutor_social', 'Tutor Goal': 'tutor_goal'}

    # outcome_mapping = {'uptake': 'uptake_change', 'eliciting': 'eliciting_change', 'talktime': 'talktime_change', 'grade': 'grade_change', 'aggregate': 'aggreagate_change'}
    outcome_mapping = {'aggregate': 'aggregate_change'}

    covariates = ["attendance_1", "rating_1", "talktime_tutor_pct_1", "spoken_token_tutor_pct_1", "chat_token_tutor_pct_1",
                  "length_utterance_tutor_1", "length_utterance_student_1", "length_utterance_tutor_chat_1", 
                  "length_utterance_student_chat_1", "ratio_students_engaged_1",
                  "normalized_num_turns_1", "normalized_num_high_uptakes_1", "normalized_num_eliciting_1", 
                  "normalized_num_questions_students_1", "normalized_num_questions_tutor_1", "normalized_student_reasoning_1", 
                  "min_sat_score_series", "max_sat_score_series", "grade_for_session_1"]

    # outcome_measure = outcome_mapping[curr_outcome]
    # arm = arm_mapping[curr_arm]

    # 2. SET HYPERPARAMETERS + LOAD IN DATA
    num_epochs = 400
    input_size = 19
    hidden_size = 10
    output_size = 1
    dropout_prob = 0.1

    for arm in arm_mapping.values():
        for outcome_name in outcome_mapping.keys():
            outcome_measure = outcome_mapping[outcome_name]
            best_path = f"models/{arm}_{outcome_measure}_model.pth"

            model = PolicyPredictor(input_size, hidden_size, output_size, dropout_prob)
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            X_train, y_train, X_test, y_test = load_data(arm, covariates, outcome_measure)
            train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test)

            # 4. TRAIN MODEL
            print(f"Training {arm} + {outcome_name} model")
            train_model(model, train_loader, test_loader, loss_fn, optimizer, num_epochs, best_path)

            # Load the best model and make predictions
            model.load_state_dict(torch.load(best_path))
            model.eval()
            with torch.no_grad():
                predictions = model(X_test)

if __name__ == "__main__":
    main()

