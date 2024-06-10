# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from scipy.stats import f_oneway

# Load the data
file_path = 'webseitenvergleich-1.csv'
data = pd.read_csv(file_path)

# Create a clean version without the first two columns
data_clean = data.iloc[:, 2:]

# Display the first few rows of the data to understand its structure
data.head()

# Change Name of the column 'Auf welchem Gerät führen Sie dieseamfrage durch?' to 'Device'
data.rename(columns={'Auf welchem Gerät führen Sie dieseamfrage durch?': 'Gerät'}, inplace=True)
print("\n")
print("Column Index after renaming: ")
print(data.columns)

# Extract data for each approach

# Ansatz 1
inter = pd.concat([data.iloc[:, 2:8], data.iloc[:, 20:26], data.iloc[:, 38:44]], axis=1)
print("\n")
print("Inter Data: ")
print(inter)

# Ansatz 2
auto = pd.concat([data.iloc[:, 8:14], data.iloc[:, 32:38], data.iloc[:, 44:50]], axis=1)
print("\n")
print("Auto Data: ")
print(auto)

# Ansatz 3
template = pd.concat([data.iloc[:, 14:20], data.iloc[:, 26:32], data.iloc[:, 50:56]], axis=1)
print("\n")
print("Template Data: ")
print(template)

# Renaming columns for clarity (assuming each set has similar questions)
columns = ['attraktiv', 'unterstützend', 'einfach', 'übersichtlich', 'interessant', 'neuartig']

print("\n")
print("Inter Columns: ")
print(inter.columns)
inter.columns = columns * 3
print("\n")
print("Inter Columns after renaming: ")
print(inter.columns)
auto.columns = columns * 3
template.columns = columns * 3

# Display a summary of the data
inter_summary = inter.describe()
auto_summary = auto.describe()
template_summary = template.describe()

print("\n")
print("Inter Summary: ")
print(inter_summary)
print("\n")
print("Auto Summary: ")
print(auto_summary)
print("\n")
print("Template Summary: ")
print(template_summary)

# Combine the summary statistics for each approach into one table

summary_table = pd.concat([inter_summary, auto_summary, template_summary], keys=['Auto', 'Inter', 'Template'])

# Display the table and enable export into csv
print("\n")
print("Summary Table: ")
print(summary_table)
summary_table.to_csv('summary_table.csv')

# Combine similiar criteria (e.g. attraktiv, attraktiv1, attraktiv2) into one criterion for each approach, keep the order of the columns

# Combine the columns for each approach by taking the mean of the columns with the same criterion (keeps the order of the columns)
# save the order of the columns
column_order = inter.columns
# remove repeated columns from order
column_order = column_order.drop_duplicates()
# group the data and order the columns
inter_grouped = inter.groupby(level=0, axis=1).mean()
inter_grouped = inter_grouped[column_order]

auto_grouped = auto.groupby(level=0, axis=1).mean()
auto_grouped = auto_grouped[column_order]

template_grouped = template.groupby(level=0, axis=1).mean()
template_grouped = template_grouped[column_order]


# Display a summary of the grouped data
auto_grouped_summary = auto_grouped.describe()
inter_grouped_summary = inter_grouped.describe()
template_grouped_summary = template_grouped.describe()

# Combine the summary statistics for each approach into one table
grouped_summary_table = pd.concat([auto_grouped_summary, inter_grouped_summary, template_grouped_summary], keys=['Auto', 'Inter', 'Template'])

# Display the table and enable export into csv and limit the decimal places to 2

grouped_summary_table_round = grouped_summary_table.round(2)
print("\n")
print("Grouped Summary Table: ")
print(grouped_summary_table_round)
grouped_summary_table_round.to_csv('grouped_summary_table_round.csv')

# Display the table as a figure using matplotlib
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')
ax.table(cellText=grouped_summary_table_round.values, colLabels=grouped_summary_table_round.columns, rowLabels=grouped_summary_table_round.index, loc='center', cellLoc='center')
# Typography: Times New Roman 12
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.show()

# Create a plot to visualize the average scores and stds as error bars for each approach and criterion in grouped data

# Define the categories
categories = ['Attraktiv', 'Unterstützend', 'Einfach', 'Übersichtlich', 'Interessant', 'Neuartig']

# Creating the bar plot with the y axis from 1 to 7
x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 8))

# Set the y-axis from 1 to 7
ax.set_ylim(1, 7.5)

# Extract the mean and std for each approach
print("\n")
print("Auto Grouped Summary: ")
print(auto_grouped_summary)

print("\n")
print("Auto Grouped Summary Mean: ")
print(auto_grouped_summary.loc['mean'])


# Plotting the bars
bar1 = ax.bar(x - width, auto_grouped_summary.loc['mean'], width, yerr=auto_grouped_summary.loc['std'], label='Auto', capsize=5)
bar2 = ax.bar(x, inter_grouped_summary.loc['mean'], width, yerr=inter_grouped_summary.loc['std'], label='Inter', capsize=5)
bar3 = ax.bar(x + width, template_grouped_summary.loc['mean'], width, yerr=template_grouped_summary.loc['std'], label='Template', capsize=5)

# Adding labels, title, and custom x-axis tick labels
ax.set_ylabel('Punkte')
#ax.set_title('Bewertung der Ansätze in den Bewertungskriterien')
ax.set_xticks(x)
ax.set_xticklabels(categories)
# Legend
ax.legend()
# Typography: Times New Roman 12
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

fig.tight_layout()

# Display the plot
plt.show()

# # Create a box plot to visualize the average scores and stds as error bars for each approach and criterion in grouped data

# Define the categories
categories = ['Attraktiv', 'Unterstützend', 'Einfach', 'Übersichtlich', 'Interessant', 'Neuartig']

# Creating the box plot with the y axis from 1 to 7
x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 8))

# Set the y-axis from 1 to 7
ax.set_ylim(1, 7.5)

# Plotting the box plot

box1 = ax.boxplot(auto_grouped, positions=np.array(range(len(auto_grouped.columns)))-width, widths=width, patch_artist=True, meanline=True, showmeans=True, showcaps=True, boxprops=dict(facecolor='#1f77b4'), whiskerprops=dict(color='blue'), capprops=dict(color='blue'), meanprops=dict(color='black'),medianprops=dict(color='red'),label='Auto')
box2 = ax.boxplot(inter_grouped, positions=np.array(range(len(inter_grouped.columns))), widths=width, patch_artist=True, meanline=True, showmeans=True, showcaps=True, boxprops=dict(facecolor='#ff7f0e'), whiskerprops=dict(color='red'), capprops=dict(color='red'), meanprops=dict(color='black'),medianprops=dict(color='red'), label='Inter')
box3 = ax.boxplot(template_grouped, positions=np.array(range(len(template_grouped.columns)))+width, widths=width, patch_artist=True, meanline=True, showmeans=True, showcaps=True, boxprops=dict(facecolor='#2ca02c'), whiskerprops=dict(color='green'), capprops=dict(color='green'), meanprops=dict(color='black'), medianprops=dict(color='red'), label='Template')

# Adding labels, title, and custom x-axis tick labels
ax.set_ylabel('Punkte')
#ax.set_title('Punkte der Ansätze in den Bewertungskriterien')
ax.set_xticks(x)
ax.set_xticklabels(categories)  
# Add dashed meanline and solid medianline to the legend
meanline = plt.Line2D([0], [0], color='black', linewidth=2, linestyle='dashed', dashes=(2, 1))
medianline = plt.Line2D([0], [0], color='red', linewidth=2)
ax.legend([box1["boxes"][0], box2["boxes"][0], box3["boxes"][0], meanline, medianline], ['Auto', 'Inter', 'Template', 'Mittelwert', 'Median'], loc='best')

# Typography: Times New Roman 12
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

fig.tight_layout()

# Display the plot
plt.show()


# Combine similiar criteria (e.g. attraktiv, attraktiv1, attraktiv2) into one criterion for the whole data by combining the approaches

# Combine the approaches
data_grouped = pd.concat([inter_grouped, auto_grouped, template_grouped], axis=1)

# Rename the columns for clarity into e.g 'inter_attraktiv', 'inter_einfach', ... 'auto_attraktiv', 'auto_einfach', ... 'template_attraktiv', 'template_einfach
data_grouped.columns = [f'{approach}_{criterion}' for approach in ['inter', 'auto', 'template'] for criterion in columns]

#add the columns back
data_grouped = pd.concat([data.iloc[:, 0:2], data_grouped], axis=1)

print("\n")
print("Data Grouped: ")
print(data_grouped)

# Calculate Cronbach's Alpha for the categories in each approach

def calculate_cronbach_alpha(df):
    num_items = df.shape[1]
    sum_variances = df.var().sum()
    total_variance = df.sum(axis=1).var()
    cronbach_alpha = (num_items / (num_items - 1)) * (1 - (sum_variances / total_variance))
    return cronbach_alpha

# Display the Cronbach's Alpha for each collumn in each approach
cronbach_alpha_inter_attraktiv = calculate_cronbach_alpha(inter.filter(like='attraktiv'))
cronbach_alpha_inter_unterstützend = calculate_cronbach_alpha(inter.filter(like='unterstützend'))
cronbach_alpha_inter_einfach = calculate_cronbach_alpha(inter.filter(like='einfach'))
cronbach_alpha_inter_übersichtlich = calculate_cronbach_alpha(inter.filter(like='übersichtlich'))
cronbach_alpha_inter_interessant = calculate_cronbach_alpha(inter.filter(like='interessant'))
cronbach_alpha_inter_neuartig = calculate_cronbach_alpha(inter.filter(like='neuartig'))

cronbach_alpha_auto_attraktiv = calculate_cronbach_alpha(auto.filter(like='attraktiv'))
cronbach_alpha_auto_unterstützend = calculate_cronbach_alpha(auto.filter(like='unterstützend'))
cronbach_alpha_auto_einfach = calculate_cronbach_alpha(auto.filter(like='einfach'))
cronbach_alpha_auto_übersichtlich = calculate_cronbach_alpha(auto.filter(like='übersichtlich'))
cronbach_alpha_auto_interessant = calculate_cronbach_alpha(auto.filter(like='interessant'))
cronbach_alpha_auto_neuartig = calculate_cronbach_alpha(auto.filter(like='neuartig'))

cronbach_alpha_template_attraktiv = calculate_cronbach_alpha(template.filter(like='attraktiv'))
cronbach_alpha_template_unterstützend = calculate_cronbach_alpha(template.filter(like='unterstützend'))
print("\n")
print("Template Unterstützend: ")
print(template.filter(like='unterstützend'))
cronbach_alpha_template_einfach = calculate_cronbach_alpha(template.filter(like='einfach'))
cronbach_alpha_template_übersichtlich = calculate_cronbach_alpha(template.filter(like='übersichtlich'))
cronbach_alpha_template_interessant = calculate_cronbach_alpha(template.filter(like='interessant'))
cronbach_alpha_template_neuartig = calculate_cronbach_alpha(template.filter(like='neuartig'))

cronbach_alpha_inter_columns = {
    'attraktiv': cronbach_alpha_inter_attraktiv,
    'unterstützend': cronbach_alpha_inter_unterstützend,
    'einfach': cronbach_alpha_inter_einfach,
    'übersichtlich': cronbach_alpha_inter_übersichtlich,
    'interessant': cronbach_alpha_inter_interessant,
    'neuartig': cronbach_alpha_inter_neuartig
}


cronbach_alpha_auto_columns = {
    'attraktiv': cronbach_alpha_auto_attraktiv,
    'unterstützend': cronbach_alpha_auto_unterstützend,
    'einfach': cronbach_alpha_auto_einfach,
    'übersichtlich': cronbach_alpha_auto_übersichtlich,
    'interessant': cronbach_alpha_auto_interessant,
    'neuartig': cronbach_alpha_auto_neuartig
}

cronbach_alpha_template_columns = {
    'attraktiv': cronbach_alpha_template_attraktiv,
    'unterstützend': cronbach_alpha_template_unterstützend,
    'einfach': cronbach_alpha_template_einfach,
    'übersichtlich': cronbach_alpha_template_übersichtlich,
    'interessant': cronbach_alpha_template_interessant,
    'neuartig': cronbach_alpha_template_neuartig
}

# Combine the Cronbach's Alpha for each category in each approach into one table
cronbach_alpha_table = pd.DataFrame([cronbach_alpha_inter_columns, cronbach_alpha_auto_columns, cronbach_alpha_template_columns], index=['Inter', 'Auto', 'Template'])

# Display the table and enable export into csv
print("\n")
print("Cronbach's Alpha for each category in each approach: ")
print(cronbach_alpha_table)
cronbach_alpha_table.to_csv('cronbach_alpha_table.csv')

# Calculate the Cronbach's Alpha for each criterion
print("\n")
print("Attraktiv Data: ")
print(data.filter(regex='attraktiv*'))
cronbach_alpha_criteria = {
    'attraktiv': calculate_cronbach_alpha(data.filter(regex='attraktiv*')),
    'unterstützend': calculate_cronbach_alpha(data.filter(regex='unterstützend*')),
    'einfach': calculate_cronbach_alpha(data.filter(regex='einfach*')),
    'übersichtlich': calculate_cronbach_alpha(data.filter(regex='übersichtlich*')),
    'interessant': calculate_cronbach_alpha(data.filter(regex='interessant*')),
    'neuartig': calculate_cronbach_alpha(data.filter(regex='neuartig*'))
}

# Combine the Cronbach's Alpha for each criterion into one table
cronbach_alpha_criteria_table = pd.DataFrame(cronbach_alpha_criteria, index=['Cronbachs Alpha'])

# Display the table and enable export into csv
print("\n")
print("Cronbach's Alpha for each criterion: ")
print(cronbach_alpha_criteria_table)
cronbach_alpha_criteria_table.to_csv('cronbach_alpha_criteria_table.csv')

# Calculate the Cronbach's Alpha for each website in each approach (Set of 6 criteria)

cronbach_alpha_website1_inter = calculate_cronbach_alpha(inter.iloc[:, 0:6])
cronbach_alpha_website2_inter = calculate_cronbach_alpha(inter.iloc[:, 6:12])
cronbach_alpha_website3_inter = calculate_cronbach_alpha(inter.iloc[:, 12:18])
print("\n")
print("Inter Website 1: ")
print(inter.iloc[:, 0:6])

cronbach_alpha_website1_auto = calculate_cronbach_alpha(auto.iloc[:, 0:6])
cronbach_alpha_website2_auto = calculate_cronbach_alpha(auto.iloc[:, 6:12])
cronbach_alpha_website3_auto = calculate_cronbach_alpha(auto.iloc[:, 12:18])

cronbach_alpha_website1_template = calculate_cronbach_alpha(template.iloc[:, 0:6])
cronbach_alpha_website2_template = calculate_cronbach_alpha(template.iloc[:, 6:12])
cronbach_alpha_website3_template = calculate_cronbach_alpha(template.iloc[:, 12:18])

# Combine the Cronbach's Alpha for each website in each approach into one table
cronbach_alpha_website_table = pd.DataFrame({
    'Inter': [cronbach_alpha_website1_inter, cronbach_alpha_website2_inter, cronbach_alpha_website3_inter],
    'Auto': [cronbach_alpha_website1_auto, cronbach_alpha_website2_auto, cronbach_alpha_website3_auto],
    'Template': [cronbach_alpha_website1_template, cronbach_alpha_website2_template, cronbach_alpha_website3_template]
}, index=['Website 1', 'Website 2', 'Website 3'])

# Display the table and enable export into csv
print("\n")
print("Cronbach's Alpha for each website in each approach: ")
print(cronbach_alpha_website_table)
cronbach_alpha_website_table.to_csv('cronbach_alpha_website_table.csv')

# Perform an ANOVA to compare the different websites in each approach

# Funktion zur Durchführung der ANOVA
def perform_anova(auto_data, inter_data, template_data):
    anova_results = {}
    
    for criterion in auto_data.keys():
        f_stat, p_value = f_oneway(auto_data[criterion], inter_data[criterion], template_data[criterion])
        anova_results[criterion] = {'F-Statistic': f_stat, 'p-value': p_value}
    
    return pd.DataFrame(anova_results).T

# Define data for each website
auto_website1 = auto.iloc[:, 0:6]
auto_website2 = auto.iloc[:, 6:12]
auto_website3 = auto.iloc[:, 12:18]

inter_website1 = inter.iloc[:, 0:6]
inter_website2 = inter.iloc[:, 6:12]
inter_website3 = inter.iloc[:, 12:18]

template_website1 = template.iloc[:, 0:6]
template_website2 = template.iloc[:, 6:12]
template_website3 = template.iloc[:, 12:18]

# Perform the ANOVA for each approach
anova_results_websites_auto = perform_anova(auto_website1, auto_website2, auto_website3)
anova_results_websites_inter = perform_anova(inter_website1, inter_website2, inter_website3)
anova_results_websites_template = perform_anova(template_website1, template_website2, template_website3)

# Combine the ANOVA results for each approach into one table
anova_results_websites = pd.concat([anova_results_websites_auto, anova_results_websites_inter, anova_results_websites_template], keys=['Auto', 'Inter', 'Template'])

# Display the table and enable export into csv
print("\n")
print("ANOVA results for each website in each approach: ")
print(anova_results_websites)
anova_results_websites.to_csv('anova_results_websites.csv')


# Calculate the Cronbach's Alpha for the whole dataset

cronbach_alpha_data = calculate_cronbach_alpha(data_clean)

# Display the Cronbach's Alpha for the whole dataset as a table and enable export into csv
print("\n")
print("Cronbach's Alpha for the whole dataset: ")
print(cronbach_alpha_data)
pd.DataFrame([cronbach_alpha_data], index=['Cronbachs Alpha']).to_csv('cronbach_alpha_data.csv')

# Calculate the Cronbach's Alpha for each approach
cronbach_alpha_inter = calculate_cronbach_alpha(inter)
cronbach_alpha_auto = calculate_cronbach_alpha(auto)
cronbach_alpha_template = calculate_cronbach_alpha(template)

# Combine the Cronbach's Alpha for each approach into one table
cronbach_alpha_approach_table = pd.DataFrame([cronbach_alpha_inter, cronbach_alpha_auto, cronbach_alpha_template], index=['Inter', 'Auto', 'Template'])

# Display the Cronbach's Alpha for each approach and enable export into csv
print("\n")
print("Cronbach's Alpha for each approach: ")
print(cronbach_alpha_approach_table)
cronbach_alpha_approach_table.to_csv('cronbach_alpha_approach_table.csv')

# Calculate if the Data is normally distributed using Shapiro-Wilk-Tests
def check_normality(df):
    normality = {
        'attraktiv': df.filter(like='attraktiv').apply(lambda x: x.dropna().shape[0] > 3).all(),
        'unterstützend': df.filter(like='unterstützend').apply(lambda x: x.dropna().shape[0] > 3).all(),
        'einfach': df.filter(like='einfach').apply(lambda x: x.dropna().shape[0] > 3).all(),
        'übersichtlich': df.filter(like='übersichtlich').apply(lambda x: x.dropna().shape[0] > 3).all(),
        'interessant': df.filter(like='interessant').apply(lambda x: x.dropna().shape[0] > 3).all(),
        'neuartig': df.filter(like='neuartig').apply(lambda x: x.dropna().shape[0] > 3).all()
    }
    return normality

# Check normality for each category in each approach
normality_inter = check_normality(inter)
normality_auto = check_normality(auto)
normality_template = check_normality(template)

# Combine the normality results for each category in each approach into one table
normality_table = pd.DataFrame([normality_inter, normality_auto, normality_template], index=['Inter', 'Auto', 'Template'])

# Display the table and enable export into csv
print("\n")
print("Normality for each category in each approach: ")
print(normality_table)
normality_table.to_csv('normality_table.csv')

# Calculate the correct average scores and stds for each criterion across all sections for each approach

def calculate_overall_averages(df):
    avg_scores = {
        'attraktiv': df.filter(like='attraktiv').mean().mean(),
        'unterstützend': df.filter(like='unterstützend').mean().mean(),
        'einfach': df.filter(like='einfach').mean().mean(),
        'übersichtlich': df.filter(like='übersichtlich').mean().mean(),
        'interessant': df.filter(like='interessant').mean().mean(),
        'neuartig': df.filter(like='neuartig').mean().mean()
    }
    return avg_scores

# Testing wether the different methods have different means. Conclusion: They do not have different means.
'''
print("\n")
print("Inter like attraktiv mean: ")
print(inter.filter(like='attraktiv').mean())
print("\n")
print("Inter like attraktiv mean mean: ")
print(inter.filter(like='attraktiv').mean().mean())
print("\n")
print("Inter like attraktiv mean(1): ")
print(inter.filter(like='attraktiv').mean(1))
print("\n")
print("Inter like attraktiv mean(1) mean: ")
print(inter.filter(like='attraktiv').mean(1).mean())
print("\n")
print("inter like attraktiv with the columns combined by appending them as extra rows, then mean: ")
inter_attraktiv, inter_attraktiv1, inter_attraktiv2 = inter.filter(like='attraktiv').iloc[:, 0], inter.filter(like='attraktiv').iloc[:, 1], inter.filter(like='attraktiv').iloc[:, 2]
inter_attraktiv = pd.concat([inter_attraktiv, inter_attraktiv1, inter_attraktiv2], ignore_index=True)
print(inter_attraktiv.mean())
'''

# Calculate overall mean scores for each approach
overall_avg_inter = calculate_overall_averages(inter)
overall_avg_auto = calculate_overall_averages(auto)
overall_avg_template = calculate_overall_averages(template)

# Testing wether the different methods have different stds. Conclusion: They do have different stds, research suggests square root of exact_pooled_variance is the best method.
'''
print("\n")
print("inter like attrativ filter: ")
print(inter.filter(like='attraktiv'))
print("\n")
print("inter like attraktiv mean: ")
print(inter.filter(like='attraktiv').mean())
print("\n")
print("inter like attraktiv std: ")
print(inter.filter(like='attraktiv').std())
print("\n")
print("inter like attraktiv std mean: ")
print(inter.filter(like='attraktiv').std().mean())
print("\n")
print("inter like attraktiv mean std: ")
print(inter.filter(like='attraktiv').mean().std())
print("\n")
print("inter like attraktiv mean(1) std: ")
print(inter.filter(like='attraktiv').mean(1).std())
print("\n")
print("inter like attraktiv with the columns combined by appending them as extra rows, then std: ")
inter_attraktiv, inter_attraktiv1, inter_attraktiv2 = inter.filter(like='attraktiv').iloc[:, 0], inter.filter(like='attraktiv').iloc[:, 1], inter.filter(like='attraktiv').iloc[:, 2]
inter_attraktiv = pd.concat([inter_attraktiv, inter_attraktiv1, inter_attraktiv2], ignore_index=True)
print(inter_attraktiv.std())

print("\n")
print("the mean of the variances plus the variance of the means of the component data sets")
print(inter.filter(like='attraktiv').std().mean()+inter.filter(like='attraktiv').mean().std())

print("\n")
print("the mean of the variances plus the variance of the means of the component data sets 2")
exact_pooled_variance = inter.filter(like='attraktiv').var().mean()+inter.filter(like='attraktiv').mean().var()
print(exact_pooled_variance)
print("\n")
print("square root of the exact pooled variance to get std")
exact_pooled_std=np.sqrt(exact_pooled_variance)
print(exact_pooled_std)
'''

# Calculate overall standard deviations for each approach
def calculate_overall_stds(df):
    std_scores = {
        'attraktiv': np.sqrt(df.filter(like='attraktiv').var().mean()+df.filter(like='attraktiv').mean().var()),
        'unterstützend': np.sqrt(df.filter(like='unterstützend').var().mean()+df.filter(like='unterstützend').mean().var()),
        'einfach': np.sqrt(df.filter(like='einfach').var().mean()+df.filter(like='einfach').mean().var()),
        'übersichtlich': np.sqrt(df.filter(like='übersichtlich').var().mean()+df.filter(like='übersichtlich').mean().var()),
        'interessant': np.sqrt(df.filter(like='interessant').var().mean()+df.filter(like='interessant').mean().var()),
        'neuartig': np.sqrt(df.filter(like='neuartig').var().mean()+df.filter(like='neuartig').mean().var())
    }
    return std_scores

overall_std_inter = calculate_overall_stds(inter)
overall_std_auto = calculate_overall_stds(auto)
overall_std_template = calculate_overall_stds(template)

# Combine the overall mean scores and stds for each approach into a table, with stds in brackets next to the means
overall_avg_table = pd.DataFrame([overall_avg_inter, overall_avg_auto, overall_avg_template], index=['Inter', 'Auto', 'Template'])
overall_avg_table = overall_avg_table.astype(str) + ' (' + pd.DataFrame([overall_std_inter, overall_std_auto, overall_std_template], index=['Inter', 'Auto', 'Template']).astype(str) + ')'

# Display the table and enable export into csv
print("\n")
print("Overall average and stds for each approach: ")
print(overall_avg_table)
overall_avg_table.to_csv('overall_avg_table.csv')

# Create a plot to visualize the average scores and stds as error bars for each criterion and approach

# Define the categories and their mean scores for each approach
categories = ['Attraktiv', 'Unterstützend', 'Einfach', 'Übersichtlich', 'Interessant', 'Neuartig']

# Creating the bar plot with the y axis from 1 to 7
x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 8))

# Set the y-axis from 1 to 7
ax.set_ylim(1, 7.5)

# Plotting the bars
bar1 = ax.bar(x - width, overall_avg_auto.values(), width, yerr=overall_std_auto.values(), label='Auto', capsize=5)
bar2 = ax.bar(x, overall_avg_inter.values(), width, yerr=overall_std_inter.values(), label='Inter', capsize=5)
bar3 = ax.bar(x + width, overall_avg_template.values(), width, yerr=overall_std_template.values(), label='Template', capsize=5)

# Adding labels, title, and custom x-axis tick labels
ax.set_ylabel('Punkte')
#ax.set_title('Bewertung der Ansätze in den Bewertungskriterien')
ax.set_xticks(x)
ax.set_xticklabels(categories)
# Legend
ax.legend()
# Typography: Times New Roman 12
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

fig.tight_layout()

# Display the plot
plt.show()

# Combine all criteria for each approach into one
inter_all = inter.mean(1)
auto_all = auto.mean(1)
template_all = template.mean(1)

print("\n")
print("Inter: ")
print(inter)
print("\n")
print("Inter All: ")
print(inter_all)

# Calculate the Standard Deviation for each approach
def calculate_approach_stds(auto, inter, template):
    std_scores = {
        'Auto': np.sqrt(auto.var().mean()+auto.mean().var()),
        'Inter': np.sqrt(inter.var().mean()+inter.mean().var()),
        'Template': np.sqrt(template.var().mean()+template.mean().var())
    }
    return std_scores

std_scores = calculate_approach_stds(auto, inter, template)

# Calculate alternative standard deviation for each approach using the std of the added columns as extra rows
def calculate_approach_stds2(auto, inter, template):
    std_scores = {
        'Auto': auto.std(),
        'Inter': inter.std(),
        'Template': template.std()
    }
    return std_scores

# Add all columns as extra rows to the first column, so at the end we have one column with all the data
auto_combined_column = pd.concat([auto.iloc[:, 0], auto.iloc[:, 1], auto.iloc[:, 2], auto.iloc[:, 3], auto.iloc[:, 4], auto.iloc[:, 5], auto.iloc[:, 6], auto.iloc[:, 7], auto.iloc[:, 8], auto.iloc[:, 9], auto.iloc[:, 10], auto.iloc[:, 11]], ignore_index=True)
inter_combined_column = pd.concat([inter.iloc[:, 0], inter.iloc[:, 1], inter.iloc[:, 2], inter.iloc[:, 3], inter.iloc[:, 4], inter.iloc[:, 5], inter.iloc[:, 6], inter.iloc[:, 7], inter.iloc[:, 8], inter.iloc[:, 9], inter.iloc[:, 10], inter.iloc[:, 11]], ignore_index=True)
template_combined_column = pd.concat([template.iloc[:, 0], template.iloc[:, 1], template.iloc[:, 2], template.iloc[:, 3], template.iloc[:, 4], template.iloc[:, 5], template.iloc[:, 6], template.iloc[:, 7], template.iloc[:, 8], template.iloc[:, 9], template.iloc[:, 10], template.iloc[:, 11]], ignore_index=True)


print("\n")
print("Auto Combined Column: ")
print(auto_combined_column)

std_scores2 = calculate_approach_stds2(auto_combined_column, inter_combined_column, template_combined_column)

print("\n")
print("Standard Deviations: ")
print(std_scores)
print("\n")
print("Standard Deviations 2: ")
print(std_scores2)

# Display a summary of the combined data
inter_all_summary = inter_all.describe()
auto_all_summary = auto_all.describe()
template_all_summary = template_all.describe()

# Combine the summary statistics for each approach with combined criteria into one table
all_summary_table = pd.concat([inter_all_summary, auto_all_summary, template_all_summary], keys=['Auto', 'Inter', 'Template'])

# Display the table and enable export into csv
print("\n")
print("Summary Table for combined criteria: ")
print(all_summary_table)
# Invert summary table for better readability
all_summary_table = all_summary_table.T
all_summary_table.to_csv('all_summary_table.csv')

# Plot the mean with std as error bars for each approach
# Define the categories
categories = ['Auto', 'Inter', 'Template']

# Creating the bar plot
x = np.arange(len(categories))
width = 0.75


# Create the figure and axes objects
fig, ax = plt.subplots(figsize=(12, 8))

# Set the y-axis from 1 to 7
ax.set_ylim(1, 7.5)

# Plotting the bars with each approach having the same color as in the previous plot
bar1 = ax.bar(x, [auto_all.mean(), inter_all.mean(), template_all.mean()], width, yerr=[std_scores['Auto'], std_scores['Inter'], std_scores['Template']], capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

# Adding labels, title, and custom x-axis tick labels
ax.set_ylabel('Punkte')
#ax.set_title('Bewertung der Ansätze')
ax.set_xticks(x)
ax.set_xticklabels(categories)
# Werte groß unten im Balken anzeigen
for i in ax.patches:
    ax.text(i.get_x() + i.get_width() / 2, 1.5, str(round(i.get_height(), 2)), ha='center', va='top', color= 'white', fontsize=22, fontweight='bold')



# Legend
ax.legend()
# Typography: Times New Roman 16
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

fig.tight_layout()

# Display the plot
plt.show()

# Plot the mean with std2 as error bars for each approach
# Define the categories
categories = ['Auto', 'Inter', 'Template']

# Creating the bar plot
x = np.arange(len(categories))
width = 0.75


# Create the figure and axes objects
fig, ax = plt.subplots(figsize=(12, 8))

# Set the y-axis from 1 to 7
ax.set_ylim(1, 7.5)

# Plotting the bars with each approach having the same color as in the previous plot
bar1 = ax.bar(x, [auto_all.mean(), inter_all.mean(), template_all.mean()], width, yerr=[std_scores2['Auto'], std_scores2['Inter'], std_scores2['Template']], capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

# Adding labels, title, and custom x-axis tick labels
ax.set_ylabel('Punkte')
#ax.set_title('Bewertung der Ansätze')
ax.set_xticks(x)
ax.set_xticklabels(categories)
# Werte groß unten im Balken anzeigen
for i in ax.patches:
    ax.text(i.get_x() + i.get_width() / 2, 1.5, str(round(i.get_height(), 2)), ha='center', va='top', color= 'white', fontsize=22, fontweight='bold')



# Legend
ax.legend()
# Typography: Times New Roman 16
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

fig.tight_layout()

# Display the plot
plt.show()

# Perform a Levene's test to determine if the variances of the approaches are statistically significantly different

def perform_levene_test(approach1, approach2):
    levene_results = {
        'attraktiv': stats.levene(approach1.filter(like='attraktiv').iloc[:, 0], approach2.filter(like='attraktiv').iloc[:, 0]),
        'unterstützend': stats.levene(approach1.filter(like='unterstützend').iloc[:, 0], approach2.filter(like='unterstützend').iloc[:, 0]),
        'einfach': stats.levene(approach1.filter(like='einfach').iloc[:, 0], approach2.filter(like='einfach').iloc[:, 0]),
        'übersichtlich': stats.levene(approach1.filter(like='übersichtlich').iloc[:, 0], approach2.filter(like='übersichtlich').iloc[:, 0]),
        'interessant': stats.levene(approach1.filter(like='interessant').iloc[:, 0], approach2.filter(like='interessant').iloc[:, 0]),
        'neuartig': stats.levene(approach1.filter(like='neuartig').iloc[:, 0], approach2.filter(like='neuartig').iloc[:, 0])
    }
    return levene_results

print("\n")
print("inter.filter(like='neuartig').iloc[:, 0], template.filter(like='neuartig').iloc[:, 0]")
print((inter.filter(like='neuartig').iloc[:, 0], template.filter(like='neuartig').iloc[:, 0]))
print("Levene's test for inter-template: ")
print(stats.levene(inter.filter(like='neuartig').iloc[:, 0], template.filter(like='neuartig').iloc[:, 0]))

print("\n")
print("auto.filter(like='neuartig').iloc[:, 0], inter.filter(like='neuartig').iloc[:, 0]")
print((auto.filter(like='neuartig').iloc[:, 0], inter.filter(like='neuartig').iloc[:, 0]))
print("Levene's test for auto-inter: ")
print(stats.levene(auto.filter(like='neuartig').iloc[:, 0], inter.filter(like='neuartig').iloc[:, 0]))



# Perform Levene's test for each category in each approach
levene_auto_inter = perform_levene_test(auto, inter)
levene_auto_template = perform_levene_test(auto, template)
levene_inter_template = perform_levene_test(inter, template)

# Combine the Levene's test results for each category in each approach into one table
levene_table = pd.DataFrame([levene_auto_inter, levene_auto_template, levene_inter_template], index=['Auto-Inter', 'Auto-Template', 'Inter-Template'])

# Display the table and enable export into csv
print("\n")
print("Levene's test results for each category in each approach: ")
print(levene_table)
levene_table.to_csv('levene_approaches_table.csv')

# Perform a t-test to determine if the differences between the approaches are statistically significant

def perform_t_test(approach1, approach2):
    t_test_results = {
        'attraktiv': ttest_ind(approach1.filter(like='attraktiv').iloc[:, 0], approach2.filter(like='attraktiv').iloc[:, 0]),
        'unterstützend': ttest_ind(approach1.filter(like='unterstützend').iloc[:, 0], approach2.filter(like='unterstützend').iloc[:, 0]),
        'einfach': ttest_ind(approach1.filter(like='einfach').iloc[:, 0], approach2.filter(like='einfach').iloc[:, 0]),
        'übersichtlich': ttest_ind(approach1.filter(like='übersichtlich').iloc[:, 0], approach2.filter(like='übersichtlich').iloc[:, 0]),
        'interessant': ttest_ind(approach1.filter(like='interessant').iloc[:, 0], approach2.filter(like='interessant').iloc[:, 0]),
        'neuartig': ttest_ind(approach1.filter(like='neuartig').iloc[:, 0], approach2.filter(like='neuartig').iloc[:, 0])
    }
    return t_test_results

# Perform t-test for each category in each approach
t_test_auto_inter = perform_t_test(auto, inter)
t_test_auto_template = perform_t_test(auto, template)
t_test_inter_template = perform_t_test(inter, template)

# save the means of the approaches, p-values and the degrees of freedom seperately
means_auto = [auto.mean()[criterion] for criterion in auto]
means_inter = [inter.mean()[criterion] for criterion in inter]
p_values_auto_inter = [t_test_auto_inter[criterion].pvalue for criterion in t_test_auto_inter]
degrees_of_freedom_auto_inter = [t_test_auto_inter[criterion].df for criterion in t_test_auto_inter]

means_template = [template.mean()[criterion] for criterion in template]
p_values_auto_template = [t_test_auto_template[criterion].pvalue for criterion in t_test_auto_template]
degrees_of_freedom_auto_template = [t_test_auto_template[criterion].df for criterion in t_test_auto_template]

p_values_inter_template = [t_test_inter_template[criterion].pvalue for criterion in t_test_inter_template]
degrees_of_freedom_inter_template = [t_test_inter_template[criterion].df for criterion in t_test_inter_template]



# Combine the t-test results for each category in each approach into one table
t_test_table = pd.DataFrame([means_auto, means_inter, means_template, p_values_auto_inter, degrees_of_freedom_auto_inter, p_values_auto_template, degrees_of_freedom_auto_template, p_values_inter_template, degrees_of_freedom_inter_template], index=['Auto Mean', 'Inter Mean', 'Template Mean', 'P-Value Auto-Inter', 'Degrees of Freedom Auto-Inter', 'P-Value Auto-Template', 'Degrees of Freedom Auto-Template', 'P-Value Inter-Template', 'Degrees of Freedom Inter-Template'])

# Display the table and enable export into csv
print("\n")
print("T-test results for each category in each approach: ")
print(t_test_table)
t_test_table.to_csv('t_test_table.csv')

# Perform a t-test to determine if the differences between the approaches per category are statistically significant
def perform_t_test_alt(approach1, approach2):
    t_test_results = {
        'attraktiv': ttest_ind(approach1.filter(like='attraktiv'), approach2.filter(like='attraktiv')),
        'unterstützend': ttest_ind(approach1.filter(like='unterstützend'), approach2.filter(like='unterstützend')),
        'einfach': ttest_ind(approach1.filter(like='einfach'), approach2.filter(like='einfach')),
        'übersichtlich': ttest_ind(approach1.filter(like='übersichtlich'), approach2.filter(like='übersichtlich')),
        'interessant': ttest_ind(approach1.filter(like='interessant'), approach2.filter(like='interessant')),
        'neuartig': ttest_ind(approach1.filter(like='neuartig'), approach2.filter(like='neuartig'))
    }
    return t_test_results

# Combine columns of the same category into one column

print("\n")
print("auto.filter(like='attraktiv'): ")
print(auto.filter(like='attraktiv'))
print("\n")
print("auto.filter(like='attraktiv') only first column but with index: ")
print(auto.filter(like='attraktiv').iloc[:, 0])

auto_category_attraktiv_column = pd.concat([auto.filter(like='attraktiv').iloc[:, 0], auto.filter(like='attraktiv').iloc[:, 1], auto.filter(like='attraktiv').iloc[:, 2]], ignore_index=True)
auto_category_unterstützend_column = pd.concat([auto.filter(like='unterstützend').iloc[:, 0], auto.filter(like='unterstützend').iloc[:, 1], auto.filter(like='unterstützend').iloc[:, 2]], ignore_index=True)
auto_category_einfach_column = pd.concat([auto.filter(like='einfach').iloc[:, 0], auto.filter(like='einfach').iloc[:, 1], auto.filter(like='einfach').iloc[:, 2]], ignore_index=True)
auto_category_übersichtlich_column = pd.concat([auto.filter(like='übersichtlich').iloc[:, 0], auto.filter(like='übersichtlich').iloc[:, 1], auto.filter(like='übersichtlich').iloc[:, 2]], ignore_index=True)
auto_category_interessant_column = pd.concat([auto.filter(like='interessant').iloc[:, 0], auto.filter(like='interessant').iloc[:, 1], auto.filter(like='interessant').iloc[:, 2]], ignore_index=True)
auto_category_neuartig_column = pd.concat([auto.filter(like='neuartig').iloc[:, 0], auto.filter(like='neuartig').iloc[:, 1], auto.filter(like='neuartig').iloc[:, 2]], ignore_index=True)

# Combine the columns into one array with the category as index (dictionary does not work)
auto_category_combined_columns = pd.DataFrame({
    'attraktiv': auto_category_attraktiv_column,
    'unterstützend': auto_category_unterstützend_column,
    'einfach': auto_category_einfach_column,
    'übersichtlich': auto_category_übersichtlich_column,
    'interessant': auto_category_interessant_column,
    'neuartig': auto_category_neuartig_column
})

inter_category_attraktiv_column = pd.concat([inter.filter(like='attraktiv').iloc[:, 0], inter.filter(like='attraktiv').iloc[:, 1], inter.filter(like='attraktiv').iloc[:, 2]], ignore_index=True)
inter_category_unterstützend_column = pd.concat([inter.filter(like='unterstützend').iloc[:, 0], inter.filter(like='unterstützend').iloc[:, 1], inter.filter(like='unterstützend').iloc[:, 2]], ignore_index=True)
inter_category_einfach_column = pd.concat([inter.filter(like='einfach').iloc[:, 0], inter.filter(like='einfach').iloc[:, 1], inter.filter(like='einfach').iloc[:, 2]], ignore_index=True)
inter_category_übersichtlich_column = pd.concat([inter.filter(like='übersichtlich').iloc[:, 0], inter.filter(like='übersichtlich').iloc[:, 1], inter.filter(like='übersichtlich').iloc[:, 2]], ignore_index=True)
inter_category_interessant_column = pd.concat([inter.filter(like='interessant').iloc[:, 0], inter.filter(like='interessant').iloc[:, 1], inter.filter(like='interessant').iloc[:, 2]], ignore_index=True)
inter_category_neuartig_column = pd.concat([inter.filter(like='neuartig').iloc[:, 0], inter.filter(like='neuartig').iloc[:, 1], inter.filter(like='neuartig').iloc[:, 2]], ignore_index=True)

inter_category_combined_columns = pd.DataFrame({
    'attraktiv': inter_category_attraktiv_column,
    'unterstützend': inter_category_unterstützend_column,
    'einfach': inter_category_einfach_column,
    'übersichtlich': inter_category_übersichtlich_column,
    'interessant': inter_category_interessant_column,
    'neuartig': inter_category_neuartig_column
})

template_category_attraktiv_column = pd.concat([template.filter(like='attraktiv').iloc[:, 0], template.filter(like='attraktiv').iloc[:, 1], template.filter(like='attraktiv').iloc[:, 2]], ignore_index=True)
template_category_unterstützend_column = pd.concat([template.filter(like='unterstützend').iloc[:, 0], template.filter(like='unterstützend').iloc[:, 1], template.filter(like='unterstützend').iloc[:, 2]], ignore_index=True)
template_category_einfach_column = pd.concat([template.filter(like='einfach').iloc[:, 0], template.filter(like='einfach').iloc[:, 1], template.filter(like='einfach').iloc[:, 2]], ignore_index=True)
template_category_übersichtlich_column = pd.concat([template.filter(like='übersichtlich').iloc[:, 0], template.filter(like='übersichtlich').iloc[:, 1], template.filter(like='übersichtlich').iloc[:, 2]], ignore_index=True)
template_category_interessant_column = pd.concat([template.filter(like='interessant').iloc[:, 0], template.filter(like='interessant').iloc[:, 1], template.filter(like='interessant').iloc[:, 2]], ignore_index=True)
template_category_neuartig_column = pd.concat([template.filter(like='neuartig').iloc[:, 0], template.filter(like='neuartig').iloc[:, 1], template.filter(like='neuartig').iloc[:, 2]], ignore_index=True)

template_category_combined_columns = pd.DataFrame({
    'attraktiv': template_category_attraktiv_column,
    'unterstützend': template_category_unterstützend_column,
    'einfach': template_category_einfach_column,
    'übersichtlich': template_category_übersichtlich_column,
    'interessant': template_category_interessant_column,
    'neuartig': template_category_neuartig_column
})


print("\n")
print("Auto Combined Column: ")
print(auto_category_combined_columns)

# Perform t-test for each category
t_test_auto_inter = perform_t_test_alt(auto_category_combined_columns, inter_category_combined_columns)
t_test_auto_template = perform_t_test_alt(auto_category_combined_columns, template_category_combined_columns)
t_test_inter_template = perform_t_test_alt(inter_category_combined_columns, template_category_combined_columns)

print("\n")
print("Auto: ")
print(t_test_auto_inter)

results_auto_inter = {
    'attraktiv': f"Statistic: {t_test_auto_inter['attraktiv'].statistic}, P-Value: {t_test_auto_inter['attraktiv'].pvalue}, Degrees of Freedom: {t_test_auto_inter['attraktiv'].df}",
    'unterstützend': f"Statistic: {t_test_auto_inter['unterstützend'].statistic}, P-Value: {t_test_auto_inter['unterstützend'].pvalue}, Degrees of Freedom: {t_test_auto_inter['unterstützend'].df}",
    'einfach': f"Statistic: {t_test_auto_inter['einfach'].statistic}, P-Value: {t_test_auto_inter['einfach'].pvalue}, Degrees of Freedom: {t_test_auto_inter['einfach'].df}",
    'übersichtlich': f"Statistic: {t_test_auto_inter['übersichtlich'].statistic}, P-Value: {t_test_auto_inter['übersichtlich'].pvalue}, Degrees of Freedom: {t_test_auto_inter['übersichtlich'].df}",
    'interessant': f"Statistic: {t_test_auto_inter['interessant'].statistic}, P-Value: {t_test_auto_inter['interessant'].pvalue}, Degrees of Freedom: {t_test_auto_inter['interessant'].df}",
    'neuartig': f"Statistic: {t_test_auto_inter['neuartig'].statistic}, P-Value: {t_test_auto_inter['neuartig'].pvalue}, Degrees of Freedom: {t_test_auto_inter['neuartig'].df}"
}

results_auto_template = {
    'attraktiv': f"Statistic: {t_test_auto_template['attraktiv'].statistic}, P-Value: {t_test_auto_template['attraktiv'].pvalue}, Degrees of Freedom: {t_test_auto_template['attraktiv'].df}",
    'unterstützend': f"Statistic: {t_test_auto_template['unterstützend'].statistic}, P-Value: {t_test_auto_template['unterstützend'].pvalue}, Degrees of Freedom: {t_test_auto_template['unterstützend'].df}",
    'einfach': f"Statistic: {t_test_auto_template['einfach'].statistic}, P-Value: {t_test_auto_template['einfach'].pvalue}, Degrees of Freedom: {t_test_auto_template['einfach'].df}",
    'übersichtlich': f"Statistic: {t_test_auto_template['übersichtlich'].statistic}, P-Value: {t_test_auto_template['übersichtlich'].pvalue}, Degrees of Freedom: {t_test_auto_template['übersichtlich'].df}",
    'interessant': f"Statistic: {t_test_auto_template['interessant'].statistic}, P-Value: {t_test_auto_template['interessant'].pvalue}, Degrees of Freedom: {t_test_auto_template['interessant'].df}",
    'neuartig': f"Statistic: {t_test_auto_template['neuartig'].statistic}, P-Value: {t_test_auto_template['neuartig'].pvalue}, Degrees of Freedom: {t_test_auto_template['neuartig'].df}"
}

results_inter_template = {
    'attraktiv': f"Statistic: {t_test_inter_template['attraktiv'].statistic}, P-Value: {t_test_inter_template['attraktiv'].pvalue}, Degrees of Freedom: {t_test_inter_template['attraktiv'].df}",
    'unterstützend': f"Statistic: {t_test_inter_template['unterstützend'].statistic}, P-Value: {t_test_inter_template['unterstützend'].pvalue}, Degrees of Freedom: {t_test_inter_template['unterstützend'].df}",
    'einfach': f"Statistic: {t_test_inter_template['einfach'].statistic}, P-Value: {t_test_inter_template['einfach'].pvalue}, Degrees of Freedom: {t_test_inter_template['einfach'].df}",
    'übersichtlich': f"Statistic: {t_test_inter_template['übersichtlich'].statistic}, P-Value: {t_test_inter_template['übersichtlich'].pvalue}, Degrees of Freedom: {t_test_inter_template['übersichtlich'].df}",
    'interessant': f"Statistic: {t_test_inter_template['interessant'].statistic}, P-Value: {t_test_inter_template['interessant'].pvalue}, Degrees of Freedom: {t_test_inter_template['interessant'].df}",
    'neuartig': f"Statistic: {t_test_inter_template['neuartig'].statistic}, P-Value: {t_test_inter_template['neuartig'].pvalue}, Degrees of Freedom: {t_test_inter_template['neuartig'].df}"
}



# Combine the t-test results for each category into one table (Include degrees of freedom) so the rows should be auto-inter, auto-template, inter-template and the columns should be the categories. Each cell should contain the Statistic, p-value and the degrees of freedom.
t_test_table = pd.DataFrame([results_auto_inter, results_auto_template, results_inter_template], index=['Auto-Inter', 'Auto-Template', 'Inter-Template'])

# Display the table and enable export into csv
print("\n")
print("T-test results for each category: ")
print(t_test_table)
t_test_table.to_csv('t_test_table_alternativ.csv')

# Perform ANOVA to compare the approaches for each category
# Funktion zur Durchführung der ANOVA
def perform_anova(auto_data, inter_data, template_data):
    anova_results = {}
    
    for criterion in auto_data.keys():
        f_stat, p_value = f_oneway(auto_data[criterion], inter_data[criterion], template_data[criterion])
        anova_results[criterion] = {'F-Statistic': f_stat, 'p-value': p_value}
    
    return pd.DataFrame(anova_results).T

# ANOVA durchführen
anova_results = perform_anova(auto_category_combined_columns, inter_category_combined_columns, template_category_combined_columns)

# Ergebnisse anzeigen
print("\n")
print("ANOVA results: ")
print(anova_results)

# Tukey HSD Test durchführen
# Daten für den Tukey-HSD-Test vorbereiten
def prepare_data_for_tukey(auto_data, inter_data, template_data, criterion):
    data = np.concatenate([auto_data[criterion], inter_data[criterion], template_data[criterion]])
    groups = ['Auto'] * len(auto_data[criterion]) + ['Inter'] * len(inter_data[criterion]) + ['Template'] * len(template_data[criterion])
    return data, groups

# Tukey-HSD-Test durchführen
def perform_tukey_hsd(auto_data, inter_data, template_data):
    tukey_results = {}
    
    for criterion in auto_data.keys():
        data, groups = prepare_data_for_tukey(auto_data, inter_data, template_data, criterion)
        tukey = pairwise_tukeyhsd(data, groups)
        tukey_results[criterion] = tukey.summary()
    
    return tukey_results

# Tukey-HSD-Tests durchführen
tukey_results = perform_tukey_hsd(auto_category_combined_columns, inter_category_combined_columns, template_category_combined_columns)

# Ergebnisse anzeigen
for criterion, result in tukey_results.items():
    print(f"Tukey HSD results for {criterion}:\n{result}\n")

# Summary of the combined data
auto_combined_column_summary = auto_combined_column.describe()
inter_combined_column_summary = inter_combined_column.describe()
template_combined_column_summary = template_combined_column.describe()

# Combine the summary statistics for each approach with combined criteria into one table
combined_column_summary_table = pd.concat([auto_combined_column_summary, inter_combined_column_summary, template_combined_column_summary], keys=['Auto', 'Inter', 'Template'])

# Display the table and enable export into csv
print("\n")
print("Summary Table for combined columns: ")
print(combined_column_summary_table)
# Invert summary table for better readability
combined_column_summary_table = combined_column_summary_table.T
combined_column_summary_table.to_csv('combined_column_summary_table.csv')


# Perform a t-test to determine if the differences between the approaches are statistically significant (alternative method using the combined columns as extra rows)

# Perform t-test for each approach
t_test_auto_inter2 = ttest_ind(auto_combined_column, inter_combined_column)
t_test_auto_template2 = ttest_ind(auto_combined_column, template_combined_column)
t_test_inter_template2 = ttest_ind(inter_combined_column, template_combined_column)

# Combine the t-test results for each approach into one table
t_test_table2 = pd.DataFrame([t_test_auto_inter2, t_test_auto_template2, t_test_inter_template2], index=['Auto-Inter', 'Auto-Template', 'Inter-Template'])

# Display the table and enable export into csv
print("\n")
print("T-test results for each approach (alt.): ")
print(t_test_table2)
t_test_table2.to_csv('t_test_table2.csv')

# Filter data for devices (only keep rows that include 'Computer/Notebook' for computer and 'Smartphone' for smartphone)
computer_full = data[data['Gerät'] == 'Computer/Notebook']
print("\n")
print("Computer Data: ")
print(computer_full)
smartphone_full = data[data['Gerät'] == 'Smartphone']

print("\n")
print("Smartphone Data: ")
print(smartphone_full)

# Combine Columns for each approach into one column

inter_computer = pd.concat([computer_full.iloc[:, 2:8], computer_full.iloc[:, 20:26], computer_full.iloc[:, 38:44]], axis=1)
inter_smartphone = pd.concat([smartphone_full.iloc[:, 2:8], smartphone_full.iloc[:, 20:26], smartphone_full.iloc[:, 38:44]],  axis=1)

auto_computer = pd.concat([computer_full.iloc[:, 8:14], computer_full.iloc[:, 32:38], computer_full.iloc[:, 44:50]],  axis=1)
auto_smartphone = pd.concat([smartphone_full.iloc[:, 8:14], smartphone_full.iloc[:, 32:38], smartphone_full.iloc[:, 44:50]], axis=1)

template_computer = pd.concat([computer_full.iloc[:, 14:20], computer_full.iloc[:, 26:32], computer_full.iloc[:, 50:56]], axis=1)
template_smartphone = pd.concat([smartphone_full.iloc[:, 14:20], smartphone_full.iloc[:, 26:32], smartphone_full.iloc[:, 50:56]], axis=1)

print("\n")
print("Inter Computer: ")
print(inter_computer)

# Combine the columns into one array

computer_combined_columns = pd.concat([computer_full.iloc[:, 2], computer_full.iloc[:, 3], computer_full.iloc[:, 4], computer_full.iloc[:, 5], computer_full.iloc[:, 6], computer_full.iloc[:, 7], computer_full.iloc[:, 8], computer_full.iloc[:, 9], computer_full.iloc[:, 10], computer_full.iloc[:, 11], computer_full.iloc[:, 12], computer_full.iloc[:, 13], computer_full.iloc[:, 14], computer_full.iloc[:, 15], computer_full.iloc[:, 16], computer_full.iloc[:, 17], computer_full.iloc[:, 18], computer_full.iloc[:, 19], computer_full.iloc[:, 20], computer_full.iloc[:, 21], computer_full.iloc[:, 22], computer_full.iloc[:, 23], computer_full.iloc[:, 24], computer_full.iloc[:, 25], computer_full.iloc[:, 26], computer_full.iloc[:, 27], computer_full.iloc[:, 28], computer_full.iloc[:, 29], computer_full.iloc[:, 30], computer_full.iloc[:, 31], computer_full.iloc[:, 32], computer_full.iloc[:, 33], computer_full.iloc[:, 34], computer_full.iloc[:, 35], computer_full.iloc[:, 36], computer_full.iloc[:, 37], computer_full.iloc[:, 38], computer_full.iloc[:, 39], computer_full.iloc[:, 40], computer_full.iloc[:, 41], computer_full.iloc[:, 42], computer_full.iloc[:, 43], computer_full.iloc[:, 44], computer_full.iloc[:, 45], computer_full.iloc[:, 46], computer_full.iloc[:, 47], computer_full.iloc[:, 48], computer_full.iloc[:, 49], computer_full.iloc[:, 50], computer_full.iloc[:, 51], computer_full.iloc[:, 52], computer_full.iloc[:, 53], computer_full.iloc[:, 54], computer_full.iloc[:, 55]], ignore_index=True)
smartphone_combined_columns = pd.concat([smartphone_full.iloc[:, 2], smartphone_full.iloc[:, 3], smartphone_full.iloc[:, 4], smartphone_full.iloc[:, 5], smartphone_full.iloc[:, 6], smartphone_full.iloc[:, 7], smartphone_full.iloc[:, 8], smartphone_full.iloc[:, 9], smartphone_full.iloc[:, 10], smartphone_full.iloc[:, 11], smartphone_full.iloc[:, 12], smartphone_full.iloc[:, 13], smartphone_full.iloc[:, 14], smartphone_full.iloc[:, 15], smartphone_full.iloc[:, 16], smartphone_full.iloc[:, 17], smartphone_full.iloc[:, 18], smartphone_full.iloc[:, 19], smartphone_full.iloc[:, 20], smartphone_full.iloc[:, 21], smartphone_full.iloc[:, 22], smartphone_full.iloc[:, 23], smartphone_full.iloc[:, 24], smartphone_full.iloc[:, 25], smartphone_full.iloc[:, 26], smartphone_full.iloc[:, 27], smartphone_full.iloc[:, 28], smartphone_full.iloc[:, 29], smartphone_full.iloc[:, 30], smartphone_full.iloc[:, 31], smartphone_full.iloc[:, 32], smartphone_full.iloc[:, 33], smartphone_full.iloc[:, 34], smartphone_full.iloc[:, 35], smartphone_full.iloc[:, 36], smartphone_full.iloc[:, 37], smartphone_full.iloc[:, 38], smartphone_full.iloc[:, 39], smartphone_full.iloc[:, 40], smartphone_full.iloc[:, 41], smartphone_full.iloc[:, 42], smartphone_full.iloc[:, 43], smartphone_full.iloc[:, 44], smartphone_full.iloc[:, 45], smartphone_full.iloc[:, 46], smartphone_full.iloc[:, 47], smartphone_full.iloc[:, 48], smartphone_full.iloc[:, 49], smartphone_full.iloc[:, 50], smartphone_full.iloc[:, 51], smartphone_full.iloc[:, 52], smartphone_full.iloc[:, 53], smartphone_full.iloc[:, 54], smartphone_full.iloc[:, 55]], ignore_index=True)

print("\n")
print("Computer Combined Columns: ")
print(computer_combined_columns)

inter_computer_combined_columns = pd.concat([inter_computer.iloc[:, 0], inter_computer.iloc[:, 1], inter_computer.iloc[:, 2], inter_computer.iloc[:, 3], inter_computer.iloc[:, 4], inter_computer.iloc[:, 5], inter_computer.iloc[:, 6], inter_computer.iloc[:, 7], inter_computer.iloc[:, 8], inter_computer.iloc[:, 9], inter_computer.iloc[:, 10], inter_computer.iloc[:, 11], inter_computer.iloc[:, 12], inter_computer.iloc[:, 13], inter_computer.iloc[:, 14], inter_computer.iloc[:, 15], inter_computer.iloc[:, 16], inter_computer.iloc[:, 17]], ignore_index=True)
inter_smartphone_combined_columns = pd.concat([inter_smartphone.iloc[:, 0], inter_smartphone.iloc[:, 1], inter_smartphone.iloc[:, 2], inter_smartphone.iloc[:, 3], inter_smartphone.iloc[:, 4], inter_smartphone.iloc[:, 5], inter_smartphone.iloc[:, 6], inter_smartphone.iloc[:, 7], inter_smartphone.iloc[:, 8], inter_smartphone.iloc[:, 9], inter_smartphone.iloc[:, 10], inter_smartphone.iloc[:, 11], inter_smartphone.iloc[:, 12], inter_smartphone.iloc[:, 13], inter_smartphone.iloc[:, 14], inter_smartphone.iloc[:, 15], inter_smartphone.iloc[:, 16], inter_smartphone.iloc[:, 17]], ignore_index=True)

auto_computer_combined_columns = pd.concat([auto_computer.iloc[:, 0], auto_computer.iloc[:, 1], auto_computer.iloc[:, 2], auto_computer.iloc[:, 3], auto_computer.iloc[:, 4], auto_computer.iloc[:, 5], auto_computer.iloc[:, 6], auto_computer.iloc[:, 7], auto_computer.iloc[:, 8], auto_computer.iloc[:, 9], auto_computer.iloc[:, 10], auto_computer.iloc[:, 11], auto_computer.iloc[:, 12], auto_computer.iloc[:, 13], auto_computer.iloc[:, 14], auto_computer.iloc[:, 15], auto_computer.iloc[:, 16], auto_computer.iloc[:, 17]],  ignore_index=True)
auto_smartphone_combined_columns = pd.concat([auto_smartphone.iloc[:, 0], auto_smartphone.iloc[:, 1], auto_smartphone.iloc[:, 2], auto_smartphone.iloc[:, 3], auto_smartphone.iloc[:, 4], auto_smartphone.iloc[:, 5], auto_smartphone.iloc[:, 6], auto_smartphone.iloc[:, 7], auto_smartphone.iloc[:, 8], auto_smartphone.iloc[:, 9], auto_smartphone.iloc[:, 10], auto_smartphone.iloc[:, 11], auto_smartphone.iloc[:, 12], auto_smartphone.iloc[:, 13], auto_smartphone.iloc[:, 14], auto_smartphone.iloc[:, 15], auto_smartphone.iloc[:, 16], auto_smartphone.iloc[:, 17]], ignore_index=True)

template_computer_combined_columns = pd.concat([template_computer.iloc[:, 0], template_computer.iloc[:, 1], template_computer.iloc[:, 2], template_computer.iloc[:, 3], template_computer.iloc[:, 4], template_computer.iloc[:, 5], template_computer.iloc[:, 6], template_computer.iloc[:, 7], template_computer.iloc[:, 8], template_computer.iloc[:, 9], template_computer.iloc[:, 10], template_computer.iloc[:, 11], template_computer.iloc[:, 12], template_computer.iloc[:, 13], template_computer.iloc[:, 14], template_computer.iloc[:, 15], template_computer.iloc[:, 16], template_computer.iloc[:, 17]], ignore_index=True)
template_smartphone_combined_columns = pd.concat([template_smartphone.iloc[:, 0], template_smartphone.iloc[:, 1], template_smartphone.iloc[:, 2], template_smartphone.iloc[:, 3], template_smartphone.iloc[:, 4], template_smartphone.iloc[:, 5], template_smartphone.iloc[:, 6], template_smartphone.iloc[:, 7], template_smartphone.iloc[:, 8], template_smartphone.iloc[:, 9], template_smartphone.iloc[:, 10], template_smartphone.iloc[:, 11], template_smartphone.iloc[:, 12], template_smartphone.iloc[:, 13], template_smartphone.iloc[:, 14], template_smartphone.iloc[:, 15], template_smartphone.iloc[:, 16], template_smartphone.iloc[:, 17]], ignore_index=True)

print("\n")
print("inter_computer_combined:  ")
print(inter_computer_combined_columns)

# Combine the columns into one array with the approach as index
computer_combined_columns2 = pd.DataFrame({'Auto': auto_computer_combined_columns, 'Inter': inter_computer_combined_columns, 'Template': template_computer_combined_columns})
smartphone_combined_columns2 = pd.DataFrame({'Auto': auto_smartphone_combined_columns, 'Inter': inter_smartphone_combined_columns, 'Template': template_smartphone_combined_columns})

# Turn computer_combined_columns and smartphone_combined_columns into 3 dataframes for each approach with computer and smartphone as index
auto_computer_smartphone_columns = pd.DataFrame({'Computer': auto_computer_combined_columns, 'Smartphone': auto_smartphone_combined_columns})
inter_computer_smartphone_columns = pd.DataFrame({'Computer': inter_computer_combined_columns, 'Smartphone': inter_smartphone_combined_columns})
template_computer_smartphone_columns = pd.DataFrame({'Computer': template_computer_combined_columns, 'Smartphone': template_smartphone_combined_columns})

# Calculate the mean for each approach between the devices Computer/Notebook and Smartphone

auto_computer_mean = auto_computer_combined_columns.mean()
auto_smartphone_mean = auto_smartphone_combined_columns.mean()

inter_computer_mean = inter_computer_combined_columns.mean()
inter_smartphone_mean = inter_smartphone_combined_columns.mean()

template_computer_mean = template_computer_combined_columns.mean()
template_smartphone_mean = template_smartphone_combined_columns.mean()

# calculate std for each approach and device
auto_computer_std = auto_computer_combined_columns.std()
auto_smartphone_std = auto_smartphone_combined_columns.std()

inter_computer_std = inter_computer_combined_columns.std()
inter_smartphone_std = inter_smartphone_combined_columns.std()

template_computer_std = template_computer_combined_columns.std()
template_smartphone_std = template_smartphone_combined_columns.std()

# Combine the mean for each approach into one table
means_computer = pd.DataFrame([auto_computer_mean, inter_computer_mean, template_computer_mean], index=['Auto', 'Inter', 'Template'])
means_smart = pd.DataFrame([auto_smartphone_mean, inter_smartphone_mean, template_smartphone_mean], index=['Auto', 'Inter', 'Template'])

# Combine the std for each approach into one table
std_computer = pd.DataFrame([auto_computer_std, inter_computer_std, template_computer_std], index=['Auto', 'Inter', 'Template'])
std_smart = pd.DataFrame([auto_smartphone_std, inter_smartphone_std, template_smartphone_std], index=['Auto', 'Inter', 'Template'])

print("\n")
print("Means Computer: ")
print(means_computer)
print("\n")
print("Means Smartphone: ")
print(means_smart)

print("\n")
print("Std Computer: ")
print(std_computer)
print("\n")
print("Std Smartphone: ")
print(std_smart)

# Plot the mean for each approach between the devices Computer/Notebook and Smartphone as a column-plot with error bars

# Define the categories and their mean scores for each approach
categories = ['Auto', 'Inter', 'Template']

# Creating the bar plot with the y axis from 1 to 7
x = np.arange(len(categories))
width = 0.4

fig, ax = plt.subplots(figsize=(12, 8))

# Set the y-axis from 1 to 7
ax.set_ylim(1, 7.5)

# Plotting the bars
bar1 = ax.bar(x - width/2, computer_combined_columns2.mean(), width, label='Computer/Notebook', yerr=computer_combined_columns2.std(), capsize=5)
bar2 = ax.bar(x + width/2, smartphone_combined_columns2.mean(), width, label='Smartphone', yerr=smartphone_combined_columns2.std(), capsize=5)

# Adding labels, title, and custom x-axis tick labels
ax.set_ylabel('Punkte')
#ax.set_title('Bewertung der Ansätze in den Bewertungskriterien')
ax.set_xticks(x)
ax.set_xticklabels(categories)
# Legend
# Add text labels with mean for each bar
for bar in bar1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, 1.5 , round(yval, 2), ha='center', va='top', color= 'white', fontsize=22, fontweight='bold')
for bar in bar2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, 1.5 , round(yval, 2), ha='center', va='top', color= 'white', fontsize=22, fontweight='bold')

ax.legend()
# Typography: Times New Roman 12
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

fig.tight_layout()

# Display the plot
plt.show()


# Plot the mean for each approach between the devices Computer/Notebook and Smartphone as a column-plot with error bars (reversed)

# Define the categories and their mean scores for each approach
categories = ['Computer/Notebook', 'Smartphone']

# Creating the bar plot with the y axis from 1 to 7
x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 8))

# Set the y-axis from 1 to 7
ax.set_ylim(1, 7.5)

# Plotting the bars
bar1 = ax.bar(x - width, auto_computer_smartphone_columns.mean(), width, label='Auto', yerr=auto_computer_smartphone_columns.std(), capsize=5)
bar2 = ax.bar(x, inter_computer_smartphone_columns.mean(), width, label='Inter', yerr=inter_computer_smartphone_columns.std(), capsize=5)
bar3 = ax.bar(x + width, template_computer_smartphone_columns.mean(), width, label='Template', yerr=template_computer_smartphone_columns.std(), capsize=5)

# Adding labels, title, and custom x-axis tick labels
ax.set_ylabel('Punkte')
#ax.set_title('Bewertung der Ansätze in den Bewertungskriterien')
ax.set_xticks(x)
ax.set_xticklabels(categories)
# Legend
ax.legend()
# Typography: Times New Roman 12
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

fig.tight_layout()

# Display the plot
plt.show()




# Perform t-test for each approach
t_test_computer_smartphone = ttest_ind(computer_combined_columns, smartphone_combined_columns)
t_test_auto_computer_smartphone = ttest_ind(auto_computer_combined_columns, auto_smartphone_combined_columns)
t_test_inter_computer_smartphone = ttest_ind(inter_computer_combined_columns, inter_smartphone_combined_columns)
t_test_template_computer_smartphone = ttest_ind(template_computer_combined_columns, template_smartphone_combined_columns)


# Combine the t-test results and mean for each approach into one table
t_test_table = pd.DataFrame([t_test_computer_smartphone, t_test_auto_computer_smartphone, t_test_inter_computer_smartphone, t_test_template_computer_smartphone], index=['Computer vs Smartphone', 'Auto Computer vs Smartphone', 'Inter Computer vs Smartphone', 'Template Computer vs Smartphone'])

# Add means of respective approaches to the table
t_test_table.loc['Computer vs Smartphone', 'Mean Computer'] = computer_combined_columns.mean()
t_test_table.loc['Computer vs Smartphone', 'Mean Smartphone'] = smartphone_combined_columns.mean()
t_test_table.loc['Auto Computer vs Smartphone', 'Mean Computer'] = auto_computer_combined_columns.mean()
t_test_table.loc['Auto Computer vs Smartphone', 'Mean Smartphone'] = auto_smartphone_combined_columns.mean()
t_test_table.loc['Inter Computer vs Smartphone', 'Mean Computer'] = inter_computer_combined_columns.mean()
t_test_table.loc['Inter Computer vs Smartphone', 'Mean Smartphone'] = inter_smartphone_combined_columns.mean()
t_test_table.loc['Template Computer vs Smartphone', 'Mean Computer'] = template_computer_combined_columns.mean()
t_test_table.loc['Template Computer vs Smartphone', 'Mean Smartphone'] = template_smartphone_combined_columns.mean()

print("\n")
print("Computer vs Smartphone Mean: ")
print(computer_full.drop(columns=['Gerät', 'Timestamp']).mean(axis=None))
print(smartphone_full.drop(columns=['Gerät', 'Timestamp']).mean(axis=None))
print("\n")
print("Auto Computer vs Smartphone Mean: ")
print(auto_computer.mean(axis=None))
print(auto_smartphone.mean(axis=None))
print("\n")
print("Inter Computer vs Smartphone Mean: ")
print(inter_computer.mean(axis=None))
print(inter_smartphone.mean(axis=None))
print("\n")
print("Template Computer vs Smartphone Mean: ")
print(template_computer.mean(axis=None))
print(template_smartphone.mean(axis=None))

# Display the table and enable export into csv
print("\n")
print("T-test results for each approach (combined columns): ")
print(t_test_table)
t_test_table.to_csv('t_test_table_combined_columns.csv')

# filter grouped data for devices
computer_grouped = data_grouped[data_grouped['Gerät'] == 'Computer/Notebook']
smartphone_grouped = data_grouped[data_grouped['Gerät'] == 'Smartphone']

# filter grouped data for devices and approaches (only keep columns that include 'inter' for inter, 'auto' for auto, and 'template' for template in index)
inter_grouped_computer = computer_grouped.filter(regex='inter*')
print("\n")
print("Inter Grouped Computer: ")
print(inter_grouped_computer)
inter_grouped_smartphone = smartphone_grouped.filter(regex='inter*')

auto_grouped_computer = computer_grouped.filter(regex='auto*')
auto_grouped_smartphone = smartphone_grouped.filter(regex='auto*')

template_grouped_computer = computer_grouped.filter(regex='template*')
template_grouped_smartphone = smartphone_grouped.filter(regex='template*')

# Drop the 'Gerät' and 'Timestamp' column as it is no longer needed
computer = computer_full.drop(columns=['Gerät', 'Timestamp'])
smartphone = smartphone_full.drop(columns=['Gerät', 'Timestamp'])

# Combine the different approaches with categories for each device into one table

device_criteria_table = pd.concat([auto_grouped_computer.mean(), inter_grouped_computer.mean(), template_grouped_computer.mean(), auto_grouped_smartphone.mean(), inter_grouped_smartphone.mean(), template_grouped_smartphone.mean()], axis=1)

# Display the table and enable export into csv
print("\n")
print("Device Criteria Table: ")
print(device_criteria_table)
device_criteria_table.to_csv('device_criteria_table.csv')

# Calculate the Cronbach's Alpha for each device
cronbach_alpha_computer = calculate_cronbach_alpha(computer)
cronbach_alpha_smartphone = calculate_cronbach_alpha(smartphone)

print("\n")
print("Cronbach's Alpha for computer: ")
print(cronbach_alpha_computer)
print("\n")
print("Cronbach's Alpha for smartphone: ")
print(cronbach_alpha_smartphone)

# Check normality for each criterion for each device
normality_computer = check_normality(computer)
normality_smartphone = check_normality(smartphone)

# Combine the normality results for each criterion in each device into one table
normality_device_table = pd.DataFrame([normality_computer, normality_smartphone], index=['Computer', 'Smartphone'])

# Display the table and enable export into csv
print("\n")
print("Normality for each criterion in each device: ")
print(normality_device_table)
normality_device_table.to_csv('normality_device_table.csv')

# Perform t-test to determine if the differences in each approach between the devices Computer/Notebook and Smartphone are statistically significant

def perform_t_test_across_devices_approach(device1, device2):
    t_test_results = {
        'Auto': ttest_ind(device1.filter(regex='auto*').mean(1), device2.filter(regex='auto*').mean(1)).pvalue,
        'Inter': ttest_ind(device1.filter(regex='inter*').mean(1), device2.filter(regex='inter*').mean(1)).pvalue,
        'Template': ttest_ind(device1.filter(regex='template*').mean(1), device2.filter(regex='template*').mean(1)).pvalue
    }
    return t_test_results

t_test_computer_smartphone = perform_t_test_across_devices_approach(computer_grouped, smartphone_grouped)

# Combine the t-test results for each approach between the devices Computer/Notebook and Smartphone into one table
t_test_device_approach_table = pd.DataFrame([t_test_computer_smartphone], index=['Computer vs Smartphone'])

# Display the table and enable export into csv
print("\n")
print("T-test results comparing devices for each approach: ")
print(t_test_device_approach_table)
t_test_device_approach_table.to_csv('t_test_device_approach_table.csv')

# Alternative t-Test for checking the resullts of the previous t-test

def perform_t_test_across_devices_approach2(device1, device2):
    t_test_results = {
        'Auto': ttest_ind(device1, device2),
        'Inter': ttest_ind(device1, device2),
        'Template': ttest_ind(device1, device2)
    }
    return t_test_results

# Perform a t-test to determine if the differences in each category between the devices Computer/Notebook and Smartphone are statistically significant


def perform_t_test_across_criterion(device1, device2):
    t_test_results = {
        'attraktiv': ttest_ind(device1.filter(like='attraktiv').iloc[:, 0], device2.filter(like='attraktiv').iloc[:, 0]).pvalue,
        'unterstützend': ttest_ind(device1.filter(like='unterstützend').iloc[:, 0], device2.filter(like='unterstützend').iloc[:, 0]).pvalue,
        'einfach': ttest_ind(device1.filter(like='einfach').iloc[:, 0], device2.filter(like='einfach').iloc[:, 0]).pvalue,
        'übersichtlich': ttest_ind(device1.filter(like='übersichtlich').iloc[:, 0], device2.filter(like='übersichtlich').iloc[:, 0]).pvalue,
        'interessant': ttest_ind(device1.filter(like='interessant').iloc[:, 0], device2.filter(like='interessant').iloc[:, 0]).pvalue,
        'neuartig': ttest_ind(device1.filter(like='neuartig').iloc[:, 0], device2.filter(like='neuartig').iloc[:, 0]).pvalue
    }
    return t_test_results

# Perform t-test comparing the devices for each criterion
t_test_inter_computer_smartphone = perform_t_test_across_criterion(inter_grouped_computer, inter_grouped_smartphone)
t_test_auto_computer_smartphone = perform_t_test_across_criterion(auto_grouped_computer, auto_grouped_smartphone)
t_test_template_computer_smartphone = perform_t_test_across_criterion(template_grouped_computer, template_grouped_smartphone)

# Combine the t-test results for each criterion into one table
t_test_device_table = pd.DataFrame([t_test_inter_computer_smartphone, t_test_auto_computer_smartphone, t_test_template_computer_smartphone], index=['Inter', 'Auto', 'Template'])

# Display the table and enable export into csv
print("\n")
print("T-test results comparing devices for each criterion: ")
print(t_test_device_table)
t_test_device_table.to_csv('t_test_device_table.csv')

# Create a plot to visualize the resuilts of the t-test comparing the devices

# Define the categories and their p-values for each approach
categories = ['Attraktiv', 'Unterstützend', 'Einfach', 'Übersichtlich', 'Interessant', 'Neuartig']

# Creating the bar plot with the y axis from 0 to 1
x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 8))

# Plotting the bars
bar1 = ax.bar(x - width, list(t_test_auto_computer_smartphone.values()), width, label='Auto', capsize=5)
bar2 = ax.bar(x, list(t_test_inter_computer_smartphone.values()), width, label='Inter', capsize=5)
bar3 = ax.bar(x + width, list(t_test_template_computer_smartphone.values()), width, label='Template', capsize=5)

# Adding labels, title, and custom x-axis tick labels
ax.set_ylabel('p-Wert')
#ax.set_title('p-Werte der t-Tests zwischen den Geräten Computer vs Smartphone in den Bewertungskriterien aufgeteilt nach Ansatz')
ax.set_xticks(x)
ax.set_xticklabels(categories)
# Legend
ax.legend()
ax.set_ylim(0, 1)
# Typography: Times New Roman 12
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

fig.tight_layout()

# Display the plot
plt.show()

# Perform t-test comparing the devices for each approach and criterion (alternative approach)

# List of criteria
criteria = ['attraktiv', 'unterstützend', 'einfach', 'übersichtlich', 'interessant', 'neuartig']

# Function to perform t-tests for each method and criterion combination
def perform_t_tests(df):
    methods = ['inter', 'auto', 'template']
    results = []

    for method in methods:
        for criterion in criteria:
            # Filter the data to only include the columns for the current method and the Gerät column using filter and regex
            method_df = df.filter(regex=f'{method}*|Gerät')                     
            print("\n")
            print("Method Dataframe: ")
            print(method_df)
            computer_ratings = method_df[method_df['Gerät'] == 'Computer/Notebook'][f'{method}_{criterion}']
            smartphone_ratings = method_df[method_df['Gerät'] == 'Smartphone'][f'{method}_{criterion}']
            
            print("\n")
            print("Method: ", method)
            print("Criterion: ", criterion)
            print("Computer Ratings: ")
            print(computer_ratings)
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(computer_ratings, smartphone_ratings, nan_policy='omit')

            results.append({
                'method': method,
                'criterion': criterion,
                't_stat': t_stat,
                'p_value': p_value
            })

    results_df = pd.DataFrame(results)
    return results_df

# Perform t-tests and get results
t_test_results = perform_t_tests(data_grouped)

# Create table with results and export to csv
t_test_results.to_csv('t_test_results.csv')

# Display the table
print("\n")
print("T-test results for each approach and criterion: ")
print(t_test_results)

# Create plot to visualize the results of the t-test comparing the devices for each approach and criterion

# Define the categories and their p-values for each approach
categories = ['Attraktiv', 'Unterstützend', 'Einfach', 'Übersichtlich', 'Interessant', 'Neuartig']

# Creating the bar plot with the y axis from 0 to 0.05
x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 8))

# Plotting the bars
bar1 = ax.bar(x - width, t_test_results[t_test_results['method'] == 'auto']['p_value'], width, label='Auto', capsize=5)
bar2 = ax.bar(x, t_test_results[t_test_results['method'] == 'inter']['p_value'], width, label='Inter', capsize=5)
bar3 = ax.bar(x + width, t_test_results[t_test_results['method'] == 'template']['p_value'], width, label='Template', capsize=5)

# Adding labels, title, and custom x-axis tick labels
ax.set_ylabel('p-Wert')
#ax.set_title('p-Werte der t-Tests zwischen den Geräten Computer/Notebook und Smartphone in den Bewertungskriterien aufgeteilt nach Ansatz')
ax.set_xticks(x)
ax.set_xticklabels(categories)
# Legend
ax.legend()
ax.set_ylim(0, 1)
# Typography: Times New Roman 12
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

fig.tight_layout()

# Display the plot
plt.show()

# Perform t-test to determine if the differences between the approaches are statistically significant

# Perform t-test for each criterion in each device

# Combine the data for each device into dictionaries
computer_data = {
    'inter': inter_grouped_computer,
    'auto': auto_grouped_computer,
    'template': template_grouped_computer
}

smartphone_data = {
    'inter': inter_grouped_smartphone,
    'auto': auto_grouped_smartphone,
    'template': template_grouped_smartphone
}

# Calculate overall averages for each approach and each criterion in each device
overall_avg_auto_computer = calculate_overall_averages(auto_grouped_computer)
overall_avg_auto_smartphone = calculate_overall_averages(auto_grouped_smartphone)

overall_avg_inter_computer = calculate_overall_averages(inter_grouped_computer)
overall_avg_inter_smartphone = calculate_overall_averages(inter_grouped_smartphone)

overall_avg_template_computer = calculate_overall_averages(template_grouped_computer)
overall_avg_template_smartphone = calculate_overall_averages(template_grouped_smartphone)

# Calculate overall standard deviations for each approach and each criterion in each device
overall_std_auto_computer = calculate_overall_stds(auto_grouped_computer)
overall_std_auto_smartphone = calculate_overall_stds(auto_grouped_smartphone)

overall_std_inter_computer = calculate_overall_stds(inter_grouped_computer)
overall_std_inter_smartphone = calculate_overall_stds(inter_grouped_smartphone)

overall_std_template_computer = calculate_overall_stds(template_grouped_computer)
overall_std_template_smartphone = calculate_overall_stds(template_grouped_smartphone)


# Combine the overall mean scores for each approach and each criterion in each device into one table, with stds in brackets next to the means
overall_avg_device_table = pd.DataFrame([overall_avg_auto_computer, overall_avg_auto_smartphone, overall_avg_inter_computer, overall_avg_inter_smartphone, overall_avg_template_computer, overall_avg_template_smartphone], index=['Auto Computer', 'Auto Smartphone', 'Inter Computer', 'Inter Smartphone', 'Template Computer', 'Template Smartphone'])
overall_avg_device_table = overall_avg_device_table.astype(str) + ' (' + pd.DataFrame([overall_std_auto_computer, overall_std_auto_smartphone, overall_std_inter_computer, overall_std_inter_smartphone, overall_std_template_computer, overall_std_template_smartphone], index=['Auto Computer', 'Auto Smartphone', 'Inter Computer', 'Inter Smartphone', 'Template Computer', 'Template Smartphone']).astype(str) + ')'

# Display the table and enable export into csv
print("\n")
print("Overall average and stds for each device: ")
print(overall_avg_device_table)
overall_avg_device_table.to_csv('overall_avg_device_table.csv')

# Calculate the average score for each approach across all criteria in each device
overall_avg_auto_computer = auto_grouped_computer.mean().mean()
overall_avg_auto_smartphone = auto_grouped_smartphone.mean().mean()

overall_avg_inter_computer = inter_grouped_computer.mean().mean()
overall_avg_inter_smartphone = inter_grouped_smartphone.mean().mean()

overall_avg_template_computer = template_grouped_computer.mean().mean()
overall_avg_template_smartphone = template_grouped_smartphone.mean().mean()

# Combine the overall average scores for each approach in each device into one table (without criteria averages)
overall_avg_device_table = pd.DataFrame({
    'Auto': [overall_avg_auto_computer, overall_avg_auto_smartphone],
    'Inter': [overall_avg_inter_computer, overall_avg_inter_smartphone],
    'Template': [overall_avg_template_computer, overall_avg_template_smartphone]
}, index=['Computer', 'Smartphone'])
# Display the table and enable export into csv
print("\n")
print("Overall average for each approach in each device: ")
print(overall_avg_device_table)
overall_avg_device_table.to_csv('overall_avg_device_table.csv')

# Perform ANOVA to determine if the differences between the devices are statistically significant (In this case same as t-test)

# Perform ANOVA for each approach
anova_auto = stats.f_oneway(auto_grouped_computer.mean(), auto_grouped_smartphone.mean()).pvalue
anova_inter = stats.f_oneway(inter_grouped_computer.mean(), inter_grouped_smartphone.mean()).pvalue
anova_template = stats.f_oneway(template_grouped_computer.mean(), template_grouped_smartphone.mean()).pvalue

# Combine the ANOVA results for each approach into one table

anova_table = pd.DataFrame({
    'Auto': anova_auto,
    'Inter': anova_inter,
    'Template': anova_template
}, index=['p-Wert'])

# Display the table and enable export into csv
print("\n")
print("ANOVA results for each approach: ")
print(anova_table)
anova_table.to_csv('anova_devices_overall_avg.csv')

# Perform Levene-Test to determine if the variances between the devices are statistically significant

levene_auto = stats.levene(auto_grouped_computer.mean(), auto_grouped_smartphone.mean(), center='mean')
levene_inter = stats.levene(inter_grouped_computer.mean(), inter_grouped_smartphone.mean(), center='mean')
levene_template = stats.levene(template_grouped_computer.mean(), template_grouped_smartphone.mean(), center='mean')


print("\n")
print("Levene-Test for Auto: ")
print(levene_auto)

# Combine the Levene-Test results for each approach into one table
levene_table = pd.DataFrame({
    'Auto': levene_auto.pvalue,
    'Inter': levene_inter.pvalue,
    'Template': levene_template.pvalue
}, index=['p-Wert'])

# Display the table and enable export into csv
print("\n")
print("Levene test results comparing Computer vs. Smartphone for each approach: ")
print(levene_table)
levene_table.to_csv('levene_devices_overall_avg.csv')

#Perform t-test to determine if the differences between the devices are statistically significant

t_test_auto = ttest_ind(auto_grouped_computer.mean(), auto_grouped_smartphone.mean())
t_test_inter = ttest_ind(inter_grouped_computer.mean(), inter_grouped_smartphone.mean())
t_test_template = ttest_ind(template_grouped_computer.mean(), template_grouped_smartphone.mean())

# Combine the t-test results for each approach into one table
t_test_table = pd.DataFrame({
    'Auto': [t_test_auto.statistic, t_test_auto.pvalue, t_test_auto.df],
    'Inter': [t_test_inter.statistic, t_test_inter.pvalue, t_test_inter.df],
    'Template': [t_test_template.statistic, t_test_template.pvalue, t_test_template.df]
}, index=['Statistik', 'p-Wert', 'Freiheitsgrade'])

# Display the table and enable export into csv
print("\n")
print("T-test results comparing Computer vs. Smartphone for each approach: ")
print(t_test_table)
t_test_table.to_csv('t_test_devices_overall_avg.csv')


