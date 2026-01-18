
## Descriptive Statistics

<img width="271" height="110" alt="image" src="https://github.com/user-attachments/assets/55d46173-c586-48c6-9883-b834772ac4b5" />

**What the code does**  
In `retail_bigdata_analysis.py`, descriptive statistics are generated for all numerical columns:

```python
numerical_cols = df.select_dtypes(include=np.number).columns
print("Descriptive statistics for numerical features:\n")
print(df[numerical_cols].describe())
```

This selects all numeric features (e.g. `Age`, `Purchase_Amount`, `Satisfaction_Score`) and calls `.describe()` to compute:
- Count of non-null observations
- Mean (average)
- Standard deviation (spread around the mean)
- Minimum, 25th percentile, median (50%), 75th percentile, and maximum

**How to interpret the table**  
- The **mean** values show typical age, spend, and satisfaction levels.  
- The **standard deviation (std)** shows variability (e.g., a high std in `Purchase_Amount` indicates big differences in customer spending).  
- **Percentiles** reveal distribution shape and outliers, helping us to see if most customers cluster at low, medium, or high purchase levels or satisfaction scores.

Business implication: this gives a first quantitative profile of customer base and their purchasing behaviour.



## Correlation Analysis

<img width="323" height="219" alt="image" src="https://github.com/user-attachments/assets/f1655d06-e721-4979-82b5-c86af5af55e5" />

**What the code does**  
The script computes and visualizes correlations between key numeric variables:

```python
corr_matrix = df[['Age', 'Purchase_Amount', 'Satisfaction_Score']].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
```

This produces a correlation matrix using Pearson correlation and then renders it as a heatmap.

**How to interpret the heatmap**  
- Each cell shows the correlation between a pair of variables, ranging from **-1** to **+1**.  
- Values **close to +1** indicate that as one metric increases, the other tends to increase.  
- Values **close to -1** indicate an inverse relationship.  
- Values **near 0** imply little or no linear relationship.

Example use cases:
- Checks if **more satisfied customers (`Satisfaction_Score`) tend to spend more (`Purchase_Amount`)**.  
-  to See whether **age** is associated with spending or satisfaction.

This helps identify potential drivers of value or risk that can be explored more deeply in regression or segmentation.



## Crosstabs & Chi-Square Tests

<img width="365" height="94" alt="image" src="https://github.com/user-attachments/assets/a53dbc5e-e5a0-44a4-aef4-05860ec6f54f" />

**What the code does**  
The script evaluates how categorical features relate to churn (`Is_Churned`) using crosstabulations and Chi-Square tests:

```python
from scipy.stats import chi2_contingency

excluded_cols = ['Customer_ID', 'Name', 'Email', 'City', 'Country', 'Date_of_Purchase']

categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_cols = [col for col in categorical_cols if col not in excluded_cols]

print("Performing Chi-Square tests for categorical features vs. 'Is_Churned':\n")

for col in categorical_cols:
    print(f"--- Analyzing '{col}' ---")
    crosstab_table = pd.crosstab(df[col], df['Is_Churned'])
    print("Crosstabulation:")
    print(crosstab_table)

    chi2, p_value, dof, expected = chi2_contingency(crosstab_table)
    print(f"\\nChi-Square Statistic: {chi2:.2f}")
    print(f"P-value: {p_value:.3f}")
    # Interpretation based on p-value
```

**How to interpret the output**  
- The **crosstab** shows counts of churned vs. non-churned customers for each category (e.g., by `Gender`, `Product_Category`, `Payment_Method`, `Loyalty_Member`).  
- The **Chi-Square test** returns a **p-value**:  
  - **p < 0.05** → statistically significant association between that categorical feature and churn.  
  - **p ≥ 0.05** → no strong evidence of association.

Business implication: this highlights which segments (for example, specific product categories or payment methods) are more prone to churn and therefore may need targeted interventions.



## Regression Output Table (Logistic Regression for Churn)

<img width="212" height="122" alt="image" src="https://github.com/user-attachments/assets/cad6b9da-ec43-4edd-ac02-3087aafc7ab9" />

**What the code does**  
The script builds a logistic regression model to predict churn (`Is_Churned`) from customer and behavioral attributes:

```python
numerical_features = ['Age', 'Purchase_Amount', 'Satisfaction_Score']
categorical_features = ['Gender', 'Product_Category', 'Payment_Method', 'Loyalty_Member']

X_numerical = df[numerical_features]
X_categorical = pd.get_dummies(df[categorical_features], drop_first=True)
X = pd.concat([X_numerical, X_categorical], axis=1)

y = df['Is_Churned']

model = LogisticRegression(random_state=42, solver='liblinear')
model.fit(X_train, y_train)

coefficients = model.coef_[0]
feature_names = X.columns
coefficient_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
coefficient_df['Abs_Coefficient'] = abs(coefficient_df['Coefficient'])
coefficient_df = coefficient_df.sort_values(by='Abs_Coefficient', ascending=False).drop(columns=['Abs_Coefficient'])
print(coefficient_df)
```

Odds ratios and confidence intervals are then computed using `statsmodels`:

```python
X_train_const = sm.add_constant(X_train_numeric)
logit_model = sm.Logit(y_train_numeric, X_train_const)
logit_results = logit_model.fit()

conf_int = logit_results.conf_int(alpha=0.05)
odds_ratios = np.exp(logit_results.params)

print("Odds Ratios:\n", odds_ratios)
print("\n95% Confidence Intervals:\n", conf_int)
```

**How to interpret the regression table**  
- Each row is a feature (e.g. `Satisfaction_Score`, `Age`, or dummy variables for categories).  
- **Coefficient sign and magnitude**:  
  - Positive coefficient → higher values of the feature are associated with a **higher likelihood of churn**.  
  - Negative coefficient → higher values of the feature are associated with a **lower likelihood of churn** (more retention).  
  - Larger absolute values indicate stronger influence.  
- **Odds ratios** (exp of the coefficient):  
  - > 1: increases churn odds.  
  - < 1: reduces churn odds.

Business implication: this table ranks the most important drivers of churn, enabling us to focus on factors (e.g., low satisfaction or specific categories) that meaningfully move the needle.



## Elbow Method Validation (K-Means Clustering)

<img width="386" height="206" alt="image" src="https://github.com/user-attachments/assets/96a840ee-bafe-42bb-8de3-753961dee196" />

**What the code does**  
To segment customers, the script applies K-Means clustering and uses the Elbow Method to determine the optimal number of clusters:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init='auto')
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.show()
```

**How to interpret the Elbow plot**  
- X-axis: number of clusters.  
- Y-axis: **WCSS** (Within-Cluster Sum of Squares), a measure of how tightly grouped each cluster is.  
- As clusters increase, WCSS decreases but with diminishing returns.  
- The **“elbow” point** (where the curve starts to flatten) indicates a good balance between model complexity and cluster quality.

In the script, the number of clusters is then chosen (e.g., 3) based on this elbow:

```python
optimal_clusters = 3
kmeans_model = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42, n_init='auto')
kmeans_model.fit(X_scaled)
cluster_labels = kmeans_model.labels_
df['Cluster'] = cluster_labels
```

Business implication: this supports defining a manageable set of distinct customer segments for targeted marketing and retention strategies.



## Satisfaction Distribution and Customer Experience Decline

<img width="332" height="150" alt="image" src="https://github.com/user-attachments/assets/752f9908-36c8-48c5-8876-5e3a649d2646" />

**What the code does**  
The script visualizes the distribution of satisfaction scores using count plots:

```python
plt.figure(figsize=(8, 6))
sns.countplot(x='Satisfaction_Score', data=df, palette='viridis')
plt.title('Distribution of Customer Satisfaction Scores')
plt.xlabel('Satisfaction Score')
plt.ylabel('Count')
plt.show()

# Variant with hue
plt.figure(figsize=(8, 6))
sns.countplot(x='Satisfaction_Score', data=df, palette='viridis', hue='Satisfaction_Score', legend=False)
plt.title('Distribution of Customer Satisfaction Scores')
plt.xlabel('Satisfaction Score')
plt.ylabel('Count')
plt.show()
```

**How to interpret the satisfaction distribution**  
- X-axis: satisfaction score levels (e.g., 1–5).  
- Y-axis: number of customers at each level.  
- If bars are concentrated at lower scores, it indicates widespread dissatisfaction and potential future churn.  
- If bars cluster at higher scores, the overall experience is positive.

When combined with churn data, we can see whether low satisfaction segments are actually translating into higher churn.



## Visualization of the Clusters

<img width="272" height="182" alt="image" src="https://github.com/user-attachments/assets/1d9331cd-53ea-4f22-ae93-22fd9df6137b" />

**What the code does**  
After determining the number of clusters and fitting K-Means, the script attaches cluster labels back to the main dataframe:

```python
kmeans_model = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42, n_init='auto')
kmeans_model.fit(X_scaled)
cluster_labels = kmeans_model.labels_

print(f"K-Means model fitted with {optimal_clusters} clusters. Cluster labels generated.")

df['Cluster'] = cluster_labels
print("Cluster labels successfully added to the DataFrame as a new column 'Cluster'.")
print(df.head())
```


**How to interpret the cluster plot**  
- Each point is a customer.  
- Color indicates cluster membership.  
- Separate, dense groups indicate distinct behavioral/customer profiles.

Business implication: each cluster can be profiled (e.g., high-spend/high-satisfaction, low-spend/medium-satisfaction, etc.) and targeted with differentiated campaigns.



## Loyalty Status

<img width="245" height="136" alt="image" src="https://github.com/user-attachments/assets/ea58fb68-5fc1-4b40-b784-b5fcf7dd95ac" />

<img width="277" height="121" alt="image" src="https://github.com/user-attachments/assets/ba71700c-9b49-44bf-a448-b6759268b267" />

**What the code does**  
The script segments customers by loyalty program membership and extracts satisfaction for each group:

```python
loyalty_members = df[df['Loyalty_Member'] == 'Yes']
non_loyalty_members = df[df['Loyalty_Member'] == 'No']

loyalty_satisfaction_scores = loyalty_members['Satisfaction_Score']
non_loyalty_satisfaction_scores = non_loyalty_members['Satisfaction_Score']

print("Successfully separated loyalty and non-loyalty members and extracted their satisfaction scores.")
print(f"Number of loyalty members: {len(loyalty_members)}")
print(f"Number of non-loyalty members: {len(non_loyalty_members)}")
```

**How to interpret the loyalty status plots**  
- Look for differences in:
  - Average satisfaction scores by loyalty status.  
  - Proportions of churned vs. active customers by loyalty status (if visualized).  
- If loyalty members have **higher satisfaction** and **lower churn**, the loyalty program appears effective.  
- If non-members show worse metrics, it signals an opportunity to expand loyalty enrollment and tailor acquisition/retention offers.



## Additional Context from the Script

### Churn Definition

```python
df['Date_of_Purchase'] = pd.to_datetime(df['Date_of_Purchase'], format='%d/%m/%Y')
latest_purchase_date = df['Date_of_Purchase'].max()

churn_threshold_days = 90
churn_cutoff_date = latest_purchase_date - timedelta(days=churn_threshold_days)

df['Is_Churned'] = df['Date_of_Purchase'] < churn_cutoff_date
```

- A customer is tagged as **churned** if their **last purchase date is more than 90 days** before the latest purchase recorded in the dataset.

### Analysis Period Churn Rate

```python
analysis_duration_months = 6
end_of_analysis_period = latest_purchase_date
start_of_analysis_period = end_of_analysis_period - pd.DateOffset(months=analysis_duration_months)

customers_at_start = df[df['Date_of_Purchase'] <= start_of_analysis_period]['Customer_ID'].unique()
customers_at_end = df[df['Date_of_Purchase'] <= end_of_analysis_period]['Customer_ID'].unique()
customers_lost = set(customers_at_start) - set(customers_at_end)

churn_rate_period = (len(customers_lost) / len(customers_at_start)) * 100
print(f"Churn rate for the defined analysis period: {churn_rate_period:.2f}%")
```

This computes a **period-based churn rate** by tracking how many customers present at the start of the analysis window are no longer active at the end.

### Class Distribution and SMOTE

The script examines and balances the target class `Is_Churned`:

```python
print("Class distribution of 'Is_Churned' (count):")
print(df['Is_Churned'].value_counts())

print("\nClass distribution of 'Is_Churned' (percentage):")
print(df['Is_Churned'].value_counts(normalize=True) * 100)

from imblearn.over_sampling import SMOTE

# After reconstructing X and y
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y_numeric)

print("Original class distribution:")
print(y_numeric.value_counts())
print("\nResampled class distribution:")
print(y_resampled.value_counts())
```

This ensures the model is trained on a balanced dataset, improving its ability to detect churners.

### Model Performance Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Model Accuracy on the test set: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"ROC-AUC: {roc_auc:.2f}")
```

These metrics (accuracy, precision, recall, F1, ROC-AUC) quantify how well the logistic regression model predicts churn, complementing the visual and tabular analyses shown above.

