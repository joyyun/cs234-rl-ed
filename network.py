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

def make_split(df, arm_name):
    filtered_df = df[df['arm_name'] == arm_name]

    X = filtered_df[["attendance_1", "rating_1", "talktime_tutor_pct_1",
    "spoken_token_tutor_pct_1", "chat_token_tutor_pct_1", "length_utterance_tutor_1",
    "length_utterance_student_1", "length_utterance_tutor_chat_1", "length_utterance_student_chat_1",
    "ratio_students_engaged_1", "normalized_num_student_names_used_1", "normalized_num_turns_1",
    "normalized_num_high_uptakes_1", "normalized_num_eliciting_1", "normalized_num_questions_students_1",
    "normalized_num_questions_tutor_1", "normalized_student_reasoning_1", "min_sat_score_series",
    "max_sat_score_series", "grade_for_session_1", "arm_name", "eliciting_change", "talktime_change", "grade_change"]]
    y = filtered_df["uptake_change"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    return train_data, test_data

best_path = "best_uptake_model.pth"
best_loss = float('inf')
# Define the network with 20 input features, 10 hidden units, and 1 output unit
input_size = 20
hidden_size = 10
output_size = 1
dropout_prob = 0.1
model = PolicyPredictor(input_size, hidden_size, output_size, dropout_prob)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

full_df = pd.read_csv('summary_data_final.csv')
# load data in pytorch tensors


train_data, test_data = make_split(full_df, "Control")
train_data.to_csv('ctrl_train_data.csv', index=False)
test_data.to_csv('ctrl_test_data.csv', index=False)

train_data, test_data = make_split(full_df, "Tutor Feedback")
train_data.to_csv('tutor_train_data.csv', index=False)
test_data.to_csv('tutor_test_data.csv', index=False)

train_data, test_data = make_split(full_df, "Tutor Feedback + Goal-Oriented Learner Feedback")
train_data.to_csv('tutor_social_train_data.csv', index=False)
test_data.to_csv('tutor_social_data.csv', index=False)

train_data, test_data = make_split(full_df, "Tutor Feedback + Socially-Oriented Learner Feedback")
train_data.to_csv('tutor_goal_data.csv', index=False)
test_data.to_csv('tutor_goal_data.csv', index=False)



0/0

train_X = train_data[covariates]
train_y = train_data['uptake_change']

test_X = test_data[covariates]
test_y = test_data['uptake_change']



print(train_X)

X_train = torch.tensor(train_X.values).float()
y_train = torch.tensor(train_y.values).float()

X_test = torch.tensor(test_X.values).float()
y_test = torch.tensor(test_y.values).float()

#print(X_train)

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10)

# placeholder data
#X_train = torch.randn(100, 20)  # 100 samples, 20 features each
#y_train = torch.randn(100, 1)   # 100 samples, 1 target each

# Convert to PyTorch tensors
#X_train = torch.tensor(X_train, dtype=torch.float32)
#y_train = torch.tensor(y_train, dtype=torch.float32)

num_epochs = 200

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

print("training complete")
model.load_state_dict(torch.load(best_path))

model.eval()
with torch.no_grad():
    predictions = model(X_test)
    #print(X_test)
    print(predictions)
