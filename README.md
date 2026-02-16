# How Soccer Teams Come Back from Behind in Away Matches: Evidence from the English Premier League

## 1/ Project overview:
Playing away matches is never easy, as players must perform under the pressure of thousands of spectators. The challenge becomes even greater when away teams fall behind on the scoreboard. However, making a comeback in away matches is not impossible, as many matches have shown away teams overturning deficits under hostile conditions. A notable example is Manchester United’s 3–2 victory over Manchester City at the Etihad Stadium in 2018, despite trailing by two goals at halftime.

<p align="center">
  <img src="Images/maxresdefault.jpg" alt="Manchester United comeback" width="500">
</p>

In this project, I try to explore the answers to 2 key questions:

**What factors are associated with the away team's chances of making a comeback?**

- **Method:** Logistic Regression with Clustered Robust Standard Errors.

**How can the away team make a comeback, and what are the most common comeback scenarios?**

- **Method:** K-means Clustering.

The analysis focuses on English Premier League matches from **2011 to 2025**.

The full dataset can be accessed here: [English Premier League Dataset](https://datahub.io/core/english-premier-league)

## 2/ How I define a comeback:

In this project, a comeback is defined as a match where the away team is trailing at halftime but somehow manages to win in the end.

**Limitation:** By defining a comeback this way, matches where the away team is trailing only in the second half and manages to come back would not be counted. However, since this dataset does not include any features for the second half, this is the only feasible definition.

Additionally, I added a new feature to the model `home_lead_1` (The number of goals by which the away team is trailing at the end of the first half). It is important to mention this because, if the dataset is not properly filtered, this variable can create a causality paradox. Specifically, if the home team does not lead at all in the first half, `home_lead_1` is 0, and by definition, a comeback is impossible. Otherwise, it would appear that the more the away team is trailing, the higher the chance of a comeback. **=> Only keep samples where `home_lead_1` > =1.**

## 3/ What factors are associated with the away team's chances of making a comeback?

To address this question, I will try to interpret the coefficients of a logistic regression model. However, to ensure that the results are reliable, I can only perform the interpretation if the model satisfies all six assumptions below:

- The Response Variable is Binary
- The Observations are Independent
- There is No Multicollinearity Among Explanatory Variables
- There are No Extreme Outliers
- There is a Linear Relationship Between Explanatory Variables and the Logit of the Response Variable
- The Sample Size is Sufficiently Large

The features selected for interpretation are: `AST` (Away Shots on Target), `HF` (Home Fouls), `HST` (Home Shots on Target), `AF` (Away Fouls), `HC` (Home Corners), `AC` (Away Corners), `HY` (Home Yellow Cards), `AY` (Away Yellow Cards), `HR` (Home Red Cards), and `AR` (Away Red Cards), `home_lead_1` (The number of goals by which the home team is leading at halftime). The Target variable is `away_comeback` (1 if the away team makes a comeback, 0 otherwise)

```python
# Code
model = smf.logit(
    'away_comeback ~ AST+HF+HST+AF+HC+AC+HY+AY+HR+AR+home_lead_1',
    data=df1
).fit()

```

First, I fit a logistic regression model using all these variables. This model is only used to calculate the residuals.

**Assumption 1: The Response Variable is Binary**

Since the target feature `away_comeback` is binary, this assumption is satisfied.

**Assumption 2: The Observations are Independent**

To check this assumption, I use the Durbin-Watson test. If the result is approximately 2, it means there is no autocorrelation and the assumption is satisfied.

```python
# Code
dw = durbin_watson(model.resid_dev)
print(dw)

```

```python
# Output
1.9263494659755964

```

=> Based on this result, assumption 2 is satisfied.

**Assumption 3: There is No Multicollinearity Among Explanatory Variables**

I use the VIF (Variance Inflation Factor) test to check whether any features have a VIF score higher than 5.

```python
# VIF test result
Variable        VIF
0         const  45.019332
1           AST   1.176121
2            HF   1.193377
3           HST   1.328959
4            AF   1.231856
5            HC   1.250440
6            AC   1.282166
7            HY   1.264185
8            AY   1.235652
9            HR   1.029591
10           AR   1.024371
11  home_lead_1   1.091098

```
=> Since all features have VIF scores around 1, assumption 3 is satisfied.

**Assumption 4: There are No Extreme Outliers**

This is the most challenging part of the analysis so far. To identify influential points, I calculate the Cook's distance for each sample in the dataset. Typically, any points that exceed the threshold (4/n) are considered influential.

<p align="center">
  <img src="Images/Cook.png" alt="Cook" width="500">
</p

However, this dataset is imbalanced. Matches where the away team makes a comeback are rare events. Out of 1,825 matches, only 110 involve a successful away comeback. As a result, when calculating Cook's distance, all points with `away_comeback` = 1 tend to be flagged as outliers. These points are not data errors, they are genuine events. Removing them would only bias the model. I will keep all of these observations and accept that the coefficients may fluctuate substantially.

**Assumption 5: There is a Linear Relationship Between Explanatory Variables and the Logit of the Response Variable**

To check this assumption, I use the Box-Tidwell test. This test is conducted by multiplying each continuous independent feature by its logarithm and adding these interaction terms to the model. If the p-value of any interaction term is significant (< 0.05), it indicates that the linearity assumption between that feature and the log-odds of the target is violated. The model is specified as follows

<p align="center">
  <img src="Images/CodeCogsEqn (2).svg" alt="Clustered Variance">
</p>

```python
# Result
                           Logit Regression Results                           
==============================================================================
Dep. Variable:          away_comeback   No. Observations:                 1825
Model:                          Logit   Df Residuals:                     1807
Method:                           MLE   Df Model:                           17
Date:                Sat, 14 Feb 2026   Pseudo R-squ.:                  0.2354
Time:                        21:36:04   Log-Likelihood:                -317.76
converged:                       True   LL-Null:                       -415.59
Covariance Type:              cluster   LLR p-value:                 2.137e-32
===============================================================================
                  coef    std err          z      P>|z|      [0.025      0.975]
-------------------------------------------------------------------------------
Intercept      -5.2759      2.714     -1.944      0.052     -10.595       0.043
AST             1.9412      0.552      3.514      0.000       0.859       3.024
AST_log        -0.5942      0.203     -2.929      0.003      -0.992      -0.197
HST            -0.2347      0.322     -0.730      0.465      -0.865       0.396
HST_log        -0.0027      0.122     -0.022      0.982      -0.242       0.236
HF              0.6049      0.538      1.123      0.261      -0.450       1.660
HF_log         -0.1856      0.159     -1.167      0.243      -0.497       0.126
AF             -0.3918      0.558     -0.702      0.483      -1.485       0.702
AF_log          0.1015      0.163      0.622      0.534      -0.219       0.422
HC              0.5635      0.289      1.952      0.051      -0.002       1.129
HC_log         -0.2061      0.105     -1.956      0.051      -0.413       0.000
AC             -0.1768      0.277     -0.638      0.524      -0.720       0.367
AC_log          0.0540      0.092      0.589      0.556      -0.126       0.234
HY              0.0807      0.090      0.893      0.372      -0.096       0.258
AY              0.1837      0.109      1.684      0.092      -0.030       0.397
HR              1.1085      0.318      3.482      0.000       0.485       1.732
AR             -0.5693      0.373     -1.525      0.127      -1.301       0.163
home_lead_1    -1.6341      0.541     -3.020      0.003      -2.695      -0.574
===============================================================================
```

The results show that `AST_log` is the only interaction term with a p-value below 0.05. To correct for this nonlinearity, I will apply a logarithmic transformation to this variable.

**Assumption 6: The Sample Size is Sufficiently Large**

How should “large” be defined in this context? I apply a rule of thumb.

<p align="center">
  <img src="Images/CodeCogsEqn (1).svg" alt="Clustered Variance">
</p>

- **Event per Variable (EPV)**: The number of events observed for each variable in a dataset (Should be >= 10)
- **Number of events**: Total number of events with y = 1
- **Number of predictors**: Total number of variables

While EPV is exactly 10, this assumption is satisfied.

**Model Interpretation**

After checking all assumptions, I will fit a logistic regression model with clustered robust standard errors. This model has two advantages compared to the first model.

**- Clustered robust standard errors:** While the Durbin-Watson test indicates independence of observations over time, this only addresses temporal independence. Observations may also be correlated within clusters; for example, Manchester City’s outstanding performance in an away match against Chelsea might be influenced by their previous encounter, in which they also defeated the same opponent at home. Clustered robust standard errors can account for this by adjusting the covariance matrix for intra-cluster correlation.

<p align="center">
  <img src="Images/CodeCogsEqn.svg" alt="Clustered Variance">
</p>

<p align="center">
  <strong>V_cluster</strong> : The robust estimate of the variance-covariance matrix of the model coefficients.<br>
  <strong>A</strong> : The Hessian matrix or the Expected Information Matrix.<br>
  <strong>B</strong> : The Outer Product of Gradients or the Score Variance Matrix.
</p>

**-Utilize log(AST) as the predictor:** Because `AST` does not have a linear relationship with the log-odds of the target, I apply a logarithmic transformation to it to improve linearity.

<p align="center">
  <img src="Images/Model.svg" alt="Clustered Variance">
</p>

```python
# Result: Coef
                           Logit Regression Results                           
==============================================================================
Dep. Variable:          away_comeback   No. Observations:                 1825
Model:                          Logit   Df Residuals:                     1807
Method:                           MLE   Df Model:                           17
Date:                Sun, 15 Feb 2026   Pseudo R-squ.:                  0.2354
Time:                        03:03:23   Log-Likelihood:                -317.76
converged:                       True   LL-Null:                       -415.59
Covariance Type:              cluster   LLR p-value:                 2.137e-32
===============================================================================
                  coef    std err          z      P>|z|      [0.025      0.975]
-------------------------------------------------------------------------------
Intercept      -5.2759      2.714     -1.944      0.052     -10.595       0.043
AST             1.9412      0.552      3.514      0.000       0.859       3.024
AST_log        -0.5942      0.203     -2.929      0.003      -0.992      -0.197
HST            -0.2347      0.322     -0.730      0.465      -0.865       0.396
HST_log        -0.0027      0.122     -0.022      0.982      -0.242       0.236
HF              0.6049      0.538      1.123      0.261      -0.450       1.660
HF_log         -0.1856      0.159     -1.167      0.243      -0.497       0.126
AF             -0.3918      0.558     -0.702      0.483      -1.485       0.702
AF_log          0.1015      0.163      0.622      0.534      -0.219       0.422
HC              0.5635      0.289      1.952      0.051      -0.002       1.129
HC_log         -0.2061      0.105     -1.956      0.051      -0.413       0.000
AC             -0.1768      0.277     -0.638      0.524      -0.720       0.367
AC_log          0.0540      0.092      0.589      0.556      -0.126       0.234
HY              0.0807      0.090      0.893      0.372      -0.096       0.258
AY              0.1837      0.109      1.684      0.092      -0.030       0.397
HR              1.1085      0.318      3.482      0.000       0.485       1.732
AR             -0.5693      0.373     -1.525      0.127      -1.301       0.163
home_lead_1    -1.6341      0.541     -3.020      0.003      -2.695      -0.574
===============================================================================
```
```python
# Result: Marginal Effect
        Logit Marginal Effects       
=====================================
Dep. Variable:          away_comeback
Method:                          dydx
At:                           overall
===============================================================================
                 dy/dx    std err          z      P>|z|      [0.025      0.975]
-------------------------------------------------------------------------------
log_AST         0.0826      0.013      6.238      0.000       0.057       0.109
HST            -0.0117      0.002     -5.370      0.000      -0.016      -0.007
HF             -0.0011      0.001     -1.115      0.265      -0.003       0.001
AF             -0.0025      0.001     -1.678      0.093      -0.005       0.000
AC             -0.0015      0.002     -0.973      0.331      -0.004       0.002
HC              0.0009      0.002      0.392      0.695      -0.004       0.005
HY              0.0039      0.005      0.867      0.386      -0.005       0.013
AY              0.0086      0.005      1.571      0.116      -0.002       0.019
HR              0.0540      0.014      3.767      0.000       0.026       0.082
AR             -0.0281      0.019     -1.468      0.142      -0.066       0.009
home_lead_1    -0.0792      0.023     -3.381      0.001      -0.125      -0.033
===============================================================================
```
The results of the model indicate that four variables are statistically significant (p-value < 0.05), both in terms of their coefficients and marginal effects:
`log_AST`, `HST`, `HR`, `home_lead_1`

**Conclusion:**

- `home_lead_1`(Number of goals the away team is trailing by in the first half): Each time the opponent scores an additional goal in the first half, the probability of a successful comeback decreases by 7.92%. This is quite reasonable, as the more goals a team is trailing by, the lower its likelihood of making a comeback.
- `HR` (Home red cards): Each additional red card received by the opponent increases the probability of a successful comeback for the away team by 5.40%. In football, receiving an extra red card means playing with fewer players, which gives the away team a substantial advantage.
- `HST` (Home shot on target): Each additional shot on target by the home team reduces the probability of a successful comeback for the away team by 24.09%. A higher number of shots on target indicates that the home team continues to press offensively after taking the lead, which may result in scoring even more goals in the second half. This factor is the most strongly associated with an away team comeback.
- `log_AST` (Away shot on target): When the away team increases its shots on target, the probability of a comeback increases a little, around 0.08 percentage points per 1% increase, on average. The first few shots on target by the away team have a larger impact on the probability of a comeback than later ones, such as the 10th or 11th. This makes sense: early shots indicate that the team is beginning to shift its playing style and gain momentum. In contrast, a large number of shots later in the game may reflect a persistent but ineffective attack rather than a good change in strategy.

## 4/ How can the away team make a comeback, and what are the most common comeback scenarios?

To answer this question, I cluster the data based on the playing styles of both teams. The team’s playing style is represented by four features: `home_attack`, `away_attack`, `home_aggressive`, and `away_aggressive`.

- `home_attack`: Total z-scores of the features associated with home team attacking:
<p align="center">
  <img src="Images/equation.svg" alt="Clustered Variance">
</p>

- `away_attack`: Total z-scores of the features associated with away team attacking:
<p align="center">
  <img src="Images/equation (1).svg" alt="Clustered Variance">
</p>

- `home_aggressive`: Total z-scores of the features associated with home team aggressiveness:
<p align="center">
  <img src="Images/equation (2).svg" alt="Clustered Variance">
</p>

- `away_aggressive`: Total z-scores of the features associated with away team aggressiveness:
<p align="center">
  <img src="Images/equation (3).svg" alt="Clustered Variance">
</p>

After defining all features, the next step is to implement the K-means algorithm with a specific k value. The optimal k (from 2 to 10) can be found by running multiple models, each model is evaluated using three metrics: Silhouette score, Davies–Bouldin index, and Calinski–Harabasz index.

<p align="center">
  <img src="Images/kmean.png" alt="Clustered Variance">
</p>

**=> All three clustering evaluation metrics drop sharply after k = 2, indicating that that the underlying playing styles in the dataset are naturally grouped into two distinct clusters.The results show that k = 2 is the optimal number of clusters.**

**Cluster Profiles Summary**

| Cluster | Samples | Home Attack (Mean) | Away Attack (Mean) | Home Aggressive (Mean) | Away Aggressive (Mean) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 25 | -0.7462 | 1.0519 | 5.4575 | 0.5364 |
| **1** | 85 | -0.8935 | 1.3025 | -0.2136 | -0.1127 |

**- Comeback by utilizing home team's disadvantages (Cluster 0):** 

**- Comeback thanks to dominant attacking play (Cluster 1):**
  

    
  

