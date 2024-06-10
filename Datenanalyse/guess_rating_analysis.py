import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import ace as tools;

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('merged_guess_rating.csv')

# Evaluate user consistency
user_consistency = df.groupby(['User', 'Approach'])['Rating'].mean().reset_index()

# Display user consistency
print("User Consistency:")
print(df.groupby(['User', 'Approach'])['Rating'].mean())

# Calculate the mean Rating for each Guess
# Add Every row where guess is Auto into auto_guess
auto_guess = df[df['Guess'] == 'Auto']
inter_guess = df[df['Guess'] == 'Inter']
template_guess = df[df['Guess'] == 'Template']

# Calculate the mean Rating for each Guess
mean_auto = auto_guess['Rating'].mean()
mean_inter = inter_guess['Rating'].mean()
mean_template = template_guess['Rating'].mean()

# Calculate the mean Rating for each Approach
auto_approach = df[df['Approach'] == 'Auto']
inter_approach = df[df['Approach'] == 'Inter']
template_approach = df[df['Approach'] == 'Template']

# Calculate the mean Rating for each Approach
mean_auto_approach = auto_approach['Rating'].mean()
mean_inter_approach = inter_approach['Rating'].mean()
mean_template_approach = template_approach['Rating'].mean()


# Calculate the mean Ratings for each User
mean_user_auto = auto_guess.groupby('User')['Rating'].mean()
mean_user_inter = inter_guess.groupby('User')['Rating'].mean()
mean_user_template = template_guess.groupby('User')['Rating'].mean()

# Combine the mean Ratings for each User into a single DataFrame
mean_user = pd.concat([mean_user_auto, mean_user_inter, mean_user_template], axis=1)
mean_user.columns = ['Auto', 'Inter', 'Template']

print(mean_user)

# Calculate mean of each column in mean_user
mean_user_mean = mean_user.mean()
print(mean_user_mean)

# Calculate the mean of each row in mean_user
mean_user_row_mean = mean_user.mean(axis=1)
print(mean_user_row_mean)

# Calculate the standard deviation of each row in mean_user
mean_user_row_std = mean_user.std(axis=1)
print(mean_user_row_std)

mean_std_rows = mean_user_row_std.mean()
print('Mean of the standard deviations of the rows:')
print(mean_std_rows)

# Calculate the standard deviation of all ratings
std_all = df['Rating'].std()

# Compare the mean Rating by each User of the Approach they guessed with the mean Rating of the Approach they rated

# Calculate the mean Rating for each User of the Approach they guessed
user_approach_mean = df.groupby(['User', 'Approach'])['Rating'].mean().reset_index()
user_approach_mean.columns = ['User', 'Approach', 'Rating']
print('user_guess_mean')
print(user_approach_mean)

user_guess_mean = df.groupby(['User', 'Guess'])['Rating'].mean().reset_index()
user_guess_mean.columns = ['User', 'Guess', 'Rating']
print('user_guess_mean')
print(user_guess_mean)

# Add every unique user to a list
std_df_user = pd.DataFrame(df['User'].unique())
# Add a column for the standard deviation of the user
std_df_user['std'] = 0
std_df_user.columns = ['User', 'std']
print(std_df_user)

for i, row in std_df_user.iterrows():
    user = row['User']
    # Calculate the standard deviation of the ratings of the user and add it to the std column of std_df_user
    uset_std = df[(df['User'] == row['User'])]['Rating'].std()
    std_df_user.loc[std_df_user['User'] == user, 'std'] = uset_std
print(std_df_user)

std_df = df['Rating'].std()
print(df)
# Check each Rating if it is 2 standard deviations away from the mean of it's row (above or below)
outliers = []

# Second loop for main logic
for i, row in df.iterrows():
    # Print the current row for context
    print(f"Processing row {i}: {row}")
    
    # Extract variables and strip whitespace
    approach = row['Approach'].strip()
    guess = row['Guess'].strip()
    user = row['User']
    company = row['Company'].strip()
    row_rating = row['Rating']
    
    # Print extracted values
    print('User:', user)
    print('Row Rating:', row_rating)
    print('Company:', company)
    print('Approach:', approach)
    print('Guess:', guess)
    
    # Print the selection for User and Company
    user_company_selection = df[(df['User'] == user) & (df['Company'].str.strip() == company)]
    print('User row selection:', user_company_selection)
    
    # Print the values we are checking for Guess
    print(f"Checking for User: {user}, Company: {company}, Approach: {approach}")

    print(user_company_selection[(user_company_selection['Guess'].str.strip() == 'Auto')])
    
    # Print the selection for User, Company, and Guess
    rating_approach_selection = df[(df['User'] == user) & (df['Company'].str.strip() == company) & (df['Guess'].str.strip() == approach)]['Rating']
    print('Rating Approach:', rating_approach_selection)
    
    # Calculate and print the absolute difference if there are matching rows
    if not rating_approach_selection.empty:
        abs_diff = abs(row_rating - rating_approach_selection.iloc[0])
        print('Absolute Difference:', abs_diff)
    else:
        print('No matching rows found for Rating Approach')

    if df[(df['User'] == user) & (df['Company'].str.strip() == company) & (df['Guess'].str.strip() == approach)].empty == False:
        # check if difference between rating and the rating of the guess is greater than 1 (The rating of the guess is the rating of the row in df where the user is the same and the approach is the same as the guess)
        if abs(row['Rating'] - df[(df['User'] == row['User']) & (df['Company'] == row['Company']) & (df['Guess'] == row['Approach'])]['Rating'].values[0]) > 2*std_df:
            guess_rating = df[(df['User'] == row['User']) & (df['Company'] == row['Company']) & (df['Guess'] == row['Approach'])]['Rating'].values[0]
            #print('user:', row['User'], 'g:', guess, 'a:', approach, 'diff:', row['Rating'] - df[(df['User'] == row['User']) & (df['Company'] == row['Company'])]['Rating'].values[0], 'rating:', row['Rating'], 'guess rating:', df[(df['User'] == row['User']) & (df['Company'] == row['Company'])]['Rating'].values[0])
            #print(f'The User {row['User']} guessed {guess}, but correct was {row['Approach']}. The Rating of the Approach was {df[(df["User"] == row["User"]) & (df["Company"] == row["Company"]) & (df['Guess'] == row['Approach'])]['Rating'].values[0]}. The Rating of the Guess was {row["Rating"]}.')
            diff = row['Rating'] - df[(df['User'] == row['User']) & (df['Company'] == row['Company']) & (df['Guess'] == row['Approach'])]['Rating'].values[0]
            if diff > 0:
                print('The rating of the approach was higher.')
            else:
                print('The rating of the approach was lower.')
            outliers.append("User: " +str(row['User']) +", Company: " + str(row['Company'])+  ", Guess: " + str(row['Guess'])+  ", Approach: "  +str(row['Approach'])+  ", Rating: " + str(row['Rating']) + ', Guess Rating: ' + str(guess_rating)+ ", Difference: "  + str(diff))

print(outliers)
# outlier to table  
outliers_df = pd.DataFrame(outliers)
outliers_df.columns = ['Outliers']
outliers_df.to_csv('outliers.csv', index=False)

# Check for outliers in the user_guess_mean
outliers_user_guess_mean = []

for i, row in user_guess_mean.iterrows():
    # Extract variables and strip whitespace
    user = row['User']
    guess = row['Guess']
    row_rating = row['Rating']
    
    # Print extracted values
    print('User:', user)
    print('Row Rating:', row_rating)
    print('Guess:', guess)

    # The rating of the other 2 guesses of the user
    other_guesses = user_guess_mean[(user_guess_mean['User'] == user) & (user_guess_mean['Guess'] != guess)]['Rating']

    # Calculate mean of the other guesses
    other_guesses_mean = other_guesses.mean()
    print('Other Guesses Mean:', other_guesses_mean)

    # Caöciulate the absolute difference
    abs_diff = abs(row_rating - other_guesses_mean)
    print('Absolute Difference:', abs_diff)
    print(std_df_user)
    print(std_df_user['User'])
    print(user)
    print('User Standard Deviation:', std_df_user[(std_df_user['User'] == user)])
    print('User Standard Deviation:', std_df_user[(std_df_user['User'] == user)]['std'].values[0])
    if abs_diff > 1.5*std_df_user[(std_df_user['User'] == user)]['std'].values[0]:
        outliers_user_guess_mean.append("User: " +str(user) +", Guess: " + str(guess) + ", Rating: " + str(row_rating) + ', Other Guesses Mean: ' + str(other_guesses_mean)+ ", Difference: "  + str(abs_diff) + ", User Standard Deviation: " + str(std_df_user[(std_df_user['User'] == user)]['std'].values[0])+", Differenz/Std "+ str(abs_diff/std_df_user[(std_df_user['User'] == user)]['std'].values[0]))

    '''
    # Calculate and print the absolute difference
    for other_guess in other_guesses:
        abs_diff = abs(row_rating - other_guess)
        print('Absolute Difference:', abs_diff)

        if abs_diff > 2*std_df:
            outliers_user_guess_mean.append("User: " +str(user) +", Guess: " + str(guess) + ", Rating: " + str(row_rating) + ', Other Guess Rating: ' + str(other_guess)+ ", Difference: "  + str(abs_diff))
    '''
    

# to table
outliers_user_guess_mean_df = pd.DataFrame(outliers_user_guess_mean)    

#outliers_user_guess_mean_df.columns = ['Outliers']
print(outliers_user_guess_mean_df)
outliers_user_guess_mean_df.to_csv('outliers_user_guess_mean.csv', index=False)