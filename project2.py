import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv('project.csv', encoding='latin1')

data.dropna(how="all", inplace=True)

data['Year'] = data["Year"].astype(int)

medals_per_year = data.groupby('Year').size().reset_index(name='Total_Medals')
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Total_Medals', data=medals_per_year, marker='o', color='blue')
plt.title('Total Olympic Medals Awarded Per Year (1976-2008)')
plt.xlabel('Olympic Year')
plt.ylabel('Total Medals Awarded')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

medals_by_gender = data['Gender'].value_counts().reset_index()
medals_by_gender.columns = ['Gender', 'Total_Medals']
plt.figure(figsize=(8, 6))
sns.barplot(x='Gender', y='Total_Medals', data=medals_by_gender, palette={'Men': 'skyblue', 'Women': 'lightcoral'})
plt.title('Total Olympic Medals by Gender (1976-2008)')
plt.xlabel('Gender')
plt.ylabel('Total Medals')
plt.show()

gender_medal_type = data.groupby(['Gender', 'Medal']).size().unstack(fill_value=0)
plt.figure(figsize=(10, 6))
gender_medal_type.plot(kind='bar', figsize=(10, 6), colormap='Paired')
plt.title('Medal Distribution by Gender and Medal Type')
plt.xlabel('Gender')
plt.ylabel('Number of Medals')
plt.xticks(rotation=0)
plt.legend(title='Medal Type')
plt.tight_layout()
plt.show()

gender_medals_yearly = data.groupby(['Year', 'Gender']).size().unstack(fill_value=0)
plt.figure(figsize=(14, 7))
sns.lineplot(x=gender_medals_yearly.index, y='Men', data=gender_medals_yearly, marker='o', label='Men', color='blue')
sns.lineplot(x=gender_medals_yearly.index, y='Women', data=gender_medals_yearly, marker='o', label='Women', color='red')
plt.title('Trend of Olympic Medals by Gender Over Time')
plt.xlabel('Olympic Year')
plt.ylabel('Number of Medals')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()

medal_type_per_year = data.groupby(['Year', 'Medal']).size().unstack(fill_value=0)
plt.figure(figsize=(14, 7))
medal_type_per_year.plot(kind='area', stacked=True, colormap='viridis', figsize=(14, 7))
plt.title('Distribution of Gold, Silver, Bronze Medals Over Time')
plt.xlabel('Olympic Year')
plt.ylabel('Number of Medals')
plt.legend(title='Medal Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

medals_per_sport = data['Sport'].value_counts().head(10).reset_index()
medals_per_sport.columns = ['Sport', 'Total_Medals']
plt.figure(figsize=(12, 7))
sns.barplot(x='Total_Medals', y='Sport', data=medals_per_sport, palette='deep')
plt.title('Top 10 Sports by Total Medals Awarded (1976-2008)')
plt.xlabel('Total Medals')
plt.ylabel('Sport')
plt.tight_layout()
plt.show()

top_countries_total_medals = data['Country'].value_counts().head(10).reset_index()
top_countries_total_medals.columns = ['Country', 'Total_Medals']
plt.figure(figsize=(12, 7))
sns.barplot(x='Total_Medals', y='Country', data=top_countries_total_medals, palette='crest')
plt.title('Top 10 Countries by Total Olympic Medals (1976-2008)')
plt.xlabel('Total Medals')
plt.ylabel('Country')
plt.show()

gold_medals = data[data['Medal'] == 'Gold']
top_countries_gold_medals = gold_medals['Country'].value_counts().head(10).reset_index()
top_countries_gold_medals.columns = ['Country', 'Gold_Medals']
plt.figure(figsize=(12, 7))
sns.barplot(x='Gold_Medals', y='Country', data=top_countries_gold_medals, palette='YlOrRd')
plt.title('Top 10 Countries by Gold Medals (1976-2008)')
plt.xlabel('Gold Medals')
plt.ylabel('Country')
plt.show()

top_athletes_total_medals = data['Athlete'].value_counts().head(10).reset_index()
top_athletes_total_medals.columns = ['Athlete', 'Total_Medals']
plt.figure(figsize=(12, 7))
sns.barplot(x='Total_Medals', y='Athlete', data=top_athletes_total_medals, palette='mako')
plt.title('Top 10 Athletes by Total Olympic Medals (1976-2008)')
plt.xlabel('Total Medals')
plt.ylabel('Athlete')
plt.show()

top_10_sports = data['Sport'].value_counts().head(10).index.tolist()
data_top_sports = data[data['Sport'].isin(top_10_sports)]
gender_in_top_sports = data_top_sports.groupby(['Sport', 'Gender']).size().unstack(fill_value=0)
gender_in_top_sports['Total'] = gender_in_top_sports['Men'] + gender_in_top_sports['Women']
gender_in_top_sports = gender_in_top_sports.sort_values(by='Total', ascending=False).drop('Total', axis=1)

plt.figure(figsize=(14, 8))
gender_in_top_sports.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='coolwarm')
plt.title('Gender Medal Distribution in Top 10 Sports (1976-2008)')
plt.xlabel('Sport')
plt.ylabel('Number of Medals')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Gender')
plt.tight_layout()
plt.show()

gender_in_top_sports_pct = gender_in_top_sports.apply(lambda x: x / x.sum() * 100, axis=1)
plt.figure(figsize=(14, 8))
gender_in_top_sports_pct.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='coolwarm')
plt.title('Percentage Gender Medal Distribution in Top 10 Sports (1976-2008)')
plt.xlabel('Sport')
plt.ylabel('Percentage of Medals')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Gender')
plt.tight_layout()
plt.show()

print("predictive analysis")
print()

df_cleaned = data.copy()

if 'Country_Code' not in df_cleaned.columns:
    df_cleaned['Country_Code'] = df_cleaned['Country']

if 'Event_gender' not in df_cleaned.columns:
    df_cleaned['Event_gender'] = df_cleaned['Gender'].apply(lambda x: 'M' if x == 'Men' else ('W' if x == 'Women' else 'X'))

df_cleaned = df_cleaned.dropna(subset=['Medal'])

le = LabelEncoder()
df_cleaned['Country_Code'] = le.fit_transform(df_cleaned['Country_Code'])
df_cleaned['Sport'] = le.fit_transform(df_cleaned['Sport'])
df_cleaned['Gender'] = le.fit_transform(df_cleaned['Gender'])
df_cleaned['Event_gender'] = le.fit_transform(df_cleaned['Event_gender'])
df_cleaned['Medal_Encoded'] = le.fit_transform(df_cleaned['Medal'])

X = df_cleaned[['Country_Code', 'Sport', 'Gender', 'Event_gender']]
y = df_cleaned['Medal_Encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000, multi_class='auto', solver='lbfgs')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

le_medal = LabelEncoder()
le_medal.fit(df_cleaned['Medal'])
target_names = le_medal.classes_

print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

