# enhanced-hotel-cancellation-prediction
This repository presents a detailed exploration into predicting hotel booking cancellations leveraging an enriched feature set. By implementing extensive feature engineering and iterative model evaluation approach, this project unveils the complex influences on cancellation probabilities, offering insights into nuanced predictive modeling.

# Hotel Bookings Cancellation Prediction with Advanced Features

## Project Overview

This project delves into predicting hotel booking cancellations with an enriched feature set, including the novel incorporation of room type match, lead time, and various interactions. Through meticulous feature engineering and model evaluation, this study unveils the nuanced influences on cancellation probabilities.

## Data Preprocessing and Avoiding Data Leakage
Before diving into the feature engineering and model development phases, it's crucial to address two key aspects: the correct sequence of data splitting and transformation, and the measures taken to prevent data leakage. This section outlines our approach to these challenges.

### Data Leakage Prevention
Data leakage can severely impact model performance by providing the model with information it wouldn't have in real-world scenarios. We've taken measures to prevent leakage by:

* Excluding predictive information: Certain variables, such as 'reservation_status,' directly reflect the target variable and were removed from the dataset.
* Avoiding test/validation data in training: Ensuring that the model is trained exclusively on the training set prevents it from learning patterns specific to the test or validation sets.

### Splitting Before Transforming
To ensure the integrity of our model:

**Split the dataset first:** This divides the data into training, testing, and validation sets. It's essential to perform this step before any data transformation to avoid inadvertently introducing bias or leakage.
**Feature Transformation:** Transformations like one-hot encoding, scaling, and balancing (through oversampling or subsampling) are applied only after the split. This approach prevents data leakage by ensuring that transformations do not use information from the test/validation sets.

### Handling Imbalanced Data
Our project also addresses the challenge of imbalanced data, common in scenarios like fraud detection where one class significantly outnumbers the other. Our strategy involves:

* Maintaining Real-world Distribution: The test set reflects the real-world class distribution to ensure the model's performance is evaluated accurately.
* Applying Balancing Techniques on the Training Set: Techniques like oversampling or subsampling are applied post-split, solely on the training set, to avoid skewing the model's evaluation.

## Feature Engineering

### Room Type Match

- **Hypothesis**: Mismatches between reserved and assigned room types might impact cancellation decisions.
- **Findings**: Incorporating a binary indicator for room type matches resulted in a nuanced change in model performance, slightly enhancing recall at a marginal precision trade-off. This suggests that discrepancies in room expectations can influence cancellation behaviors.

### Lead Time and Categorization

- **Exploration**: Analyzed lead time both as a continuous variable and in categorized form to capture booking anticipation effects.
- **Impact**: Direct inclusion of lead time significantly improved model precision and recall, indicating its predictive value. Categorization yielded comparable results, offering an alternative representation with similar predictive capabilities.

### Feature Interactions

- Explored interactions between lead time and variables such as customer type, market segment, and arrival date month, finding that:
  - Interactions introduced modest but positive changes in predictive performance.
  - These interactions likely capture complex patterns not evident when variables are considered in isolation.

### Days in Waiting List

- **Analysis**: A majority of bookings have zero waiting days, limiting the variable's variability and predictive power.
- **Decision**: Excluded from further analysis due to its limited improvement potential on model performance.

### Previous Cancellations

- **Trial**: Tested the inclusion of prior cancellation history and its interaction with lead time.
- **Result**: While initially promising, these features did not significantly enhance the model and were ultimately excluded to maintain model clarity and avoid potential overfitting.

## Model Evaluation and Improvement

- **Room Type Match**: Showed potential in identifying cancellation instances, with a slight improvement in recall.
- **Lead Time Inclusion**: Markedly improved both precision and recall, underscoring its critical role in predicting cancellations.
- **Feature Interactions**: Yielded incremental benefits, suggesting valuable insights can be gained from exploring complex feature relationships.

## Cross-validation and Model Testing

- Detailed cross-validation and testing phases underscored the robustness of the improved model against unseen data, confirming the efficacy of the selected features and interactions.

Incorporating your detailed cross-validation findings and hyperparameter optimization process into the README will give a comprehensive overview of how you enhanced the predictive model. Here's how you can format this for your README:

---

# Advanced Model Optimization for Hotel Booking Cancellations

## Cross-validation and Hyperparameter Tuning

To refine our model for predicting hotel booking cancellations, an extensive cross-validation and hyperparameter tuning process was employed. This section outlines the methodologies and key findings from this optimization phase.

### Hyperparameter Exploration

Initially, the RandomForestClassifier's `n_estimators` parameter was varied across [10, 50, 100, 200] to identify its impact on model performance:

- The models were evaluated based on accuracy and recall metrics, across validation and testing datasets.
- The optimization showed a direct correlation between the number of estimators and the improvement in both validation and testing metrics, highlighting the importance of this parameter in model refinement.

**Findings**: The model with 200 estimators showcased the best performance, indicating its suitability for our final model.

### GridSearchCV for Hyperparameter Fine-tuning

Utilizing GridSearchCV, an exhaustive search over specified parameter values was conducted. The parameter grid explored included:

- `n_estimators`: [100, 200]
- `max_depth`: [None, 10, 20, 30]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]

**Optimization Results**:
- Best Parameters: `{'model__n_estimators': 200, 'model__max_depth': 30, 'model__min_samples_leaf': 1, 'model__min_samples_split': 2}`
- Best Cross-validated Accuracy: 0.8609

This hyperparameter tuning phase underscored the critical role of model depth, leaf complexity, and the number of estimators in achieving optimal model performance.

### Cross-validation Scores

Further cross-validation with the optimized model parameters provided a comprehensive view of the model's predictive reliability:

- **Accuracy (Mean)**: 0.8576
- **Recall (Mean)**: 0.7638

These cross-validation scores confirm the model's robustness, showcasing a consistent ability to predict cancellations accurately across different data splits.

## Enhanced Model Insights

The iterative process of hyperparameter tuning and cross-validation revealed significant insights:

- The increased number of estimators (`n_estimators`) directly contributed to the model's improved precision and recall, affirming its predictive power.
- Depth and complexity control parameters (`max_depth`, `min_samples_split`, `min_samples_leaf`) were crucial in avoiding overfitting while maintaining high predictive accuracy.
- The mean cross-validation scores illustrate the model's consistency, ensuring its reliability for practical applications in predicting hotel booking cancellations.

## Conclusion

The rigorous optimization process, characterized by hyperparameter tuning and cross-validation, has significantly enhanced our predictive model. This methodical approach ensures our model's applicability in real-world scenarios, providing valuable insights into hotel booking behaviors and cancellation likelihood.

## Notebooks

- **Bookings Development with MLflow (`hotel_bookings_development.ipynb`):** The culmination of our model's development, featuring integration with MLflow for experiment tracking. It concentrates on deploying the predictive model and detailing interactions with MLflow for optimal experiment management.

- **Main Analysis and Feature Engineering (`hotel_bookings_analysis_and_feature_engineering.ipynb`):** This notebook is the cornerstone of our project, presenting an in-depth initial analysis, feature engineering, and the construction of our model's pipeline. It highlights the comprehensive process of data extraction, cleaning, and transformation tailored for model training.

- **Data Leakage Exploration (`hotel_bookings_data_leakage_analysis.ipynb`):** A critical examination aimed at identifying potential data leakage within our dataset to ensure the model's integrity and validity by preventing any artificial inflation of performance metrics.

- **Enhanced Model Development (`hotel_bookings_enhanced_model_development.ipynb`):** An iterative focus on model enhancement through sophisticated feature engineering and hyperparameter tuning. Notable advancements include:
  - **Room Type Match:** Assessment of the effect that discrepancies between reserved and assigned room types have on the likelihood of cancellation.
  - **Lead Time Analysis:** Incorporation of lead time, both as a continuous and categorical variable, to understand booking anticipation impacts.
  - **Feature Interactions:** Exploration of how interactions between lead time and other categorical variables like customer type and market segment can inform predictions.
  - **Days in Waiting List Exploration:** Investigation into the significance of the waiting period before booking confirmation on cancellations.
  - **Previous Cancellations:** Evaluation of historical cancellation data and its interaction with lead time for predictive insights.
  - **Cross-Validation and Hyperparameter Tuning (`hotel_bookings_enhanced_model_development.ipynb`):** A detailed methodology for model validation and optimization utilizing various cross-validation techniques and hyperparameter exploration.

## Data

The project employs datasets stored within the `data` folder, with `hotel_bookings.csv` serving as the primary source of booking records and engineered features used throughout the analysis.


## Technologies Used

- **Python**, **Pandas**, **Numpy**, **Matplotlib**: For data manipulation, visualization and analysis.
- **Scikit-learn**: For model development, hyperparameter tuning (GridSearchCV and cross-validation), and evaluation metrics.

## Prerequisites

Before running this analysis, please ensure you have the following:

- Python 3.7 or later.
- Libraries: pandas, numpy, scikit-learn, matplotlib. (Installation will be handled through `requirements.txt`.)
- A code editor or Jupyter Notebook environment to run the `.ipynb` files.

---

## Installation or Setup

Before diving into the analysis, ensure your environment is properly set up. Here’s how to get started:

1. **Clone the Repository**: To get a local copy of this project, run the following command in your terminal:
   ```bash
   git clone git@github.com:claraibarzabal-portfolio/enhanced-hotel-cancellation-prediction.git
   ```

2. **Set Up a Virtual Environment** (optional but recommended): To create a virtual environment, navigate to the project directory in your terminal and run:
   ```bash
   python3 -m venv venv
   ```
   Activate the virtual environment with:
   ```bash
   # For Windows
   venv\Scripts\activate
   # For Unix or MacOS
   source venv/bin/activate
   ```

3. **Install Dependencies**: Install all required libraries using the `requirements.txt` file included in the project:
   ```bash
   pip install -r requirements.txt
   ```
4. **Notebook Execution**: Launch Jupyter Notebook or JupyterLab or Google Colab.


## Conclusion

Through the process of feature engineering, model evaluation, and optimization, this project has improved the prediction of hotel booking cancellations. Our approach underscores the importance of careful feature selection and the iterative refinement of models to enhance predictive accuracy. While this study provides valuable insights into the factors influencing hotel booking cancellations, it also highlights the complexities of predictive modeling in a real-world context. The findings serve as a foundational step towards more sophisticated analyses and applications in the hospitality industry, emphasizing the potential of data-driven strategies to inform and improve decision-making processes.

## Acknowledgements

Special thanks to the creators and maintainers of the utilized datasets and libraries, facilitating this in-depth analysis.
