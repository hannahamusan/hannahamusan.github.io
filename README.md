# Amusan Hannah Oluwaseun
### Machine Learning | Data Analysis | Agricultural Economics

I build prediction models in Python and analyse data with SQL. My projects cover housing values, passenger survival and handwritten-digit classification.

I hold a B.Tech in Agricultural Economics and am training through the Webfala Women in AI Fellowship. My agricultural background shapes the questions I want to explore: what drives costs, how inputs affect returns, and how better records can support business decisions.

[LinkedIn](https://www.linkedin.com/in/amusan-hannah-oluwaseun-384745191) · [GitHub](https://github.com/hannahamusan) · [Email](mailto:amusanhannah21@gmail.com)

## Project Work

### Estimating California Housing Values

**How well can a linear regression model estimate an area’s median house value?**

I used the California housing dataset to explore this question, examining geographic price patterns before training a regression model.

The notebook includes:
- An 80/20 training and test split.
- Feature scaling fitted on the training data.
- A scikit-learn pipeline combining StandardScaler and LinearRegression.
- Model evaluation and a plot comparing predicted and actual values.
- A prediction function that accepts housing and location features.

The saved test output reports a mean squared error of **0.56**, measured in squared target units; the target is expressed in $100,000s.

**Scope:** This model estimates area-level median values from a historical dataset. It is not a valuation tool for individual homes. The model-saving section still needs correction before the trained model can be shared separately.

**Built with:** Python, Pandas, NumPy, scikit-learn and Matplotlib.

[Explore the housing notebook →](https://github.com/hannahamusan/House_price_prediction_model)

### Modelling Titanic Passenger Survival

**Using passenger information to predict survival.**

I trained logistic regression, decision tree and random forest classifiers using passenger attributes such as age, sex and class.

For the random forest, I searched **12 combinations of tree count and maximum depth** using GridSearchCV, five-fold cross-validation and accuracy scoring.

**Repository status:** The complete training and tuning work is in my local notebook. The public repository currently contains an earlier version and is awaiting an update.

**Built with:** Python, Pandas and scikit-learn.

[View the current Titanic repository →](https://github.com/hannahamusan/Titanic_data)

### Recognising Handwritten Digits

**Classifying digit images from 0 to 9.**

I trained multilayer perceptron (MLP) and convolutional neural network (CNN) models using PyTorch.

