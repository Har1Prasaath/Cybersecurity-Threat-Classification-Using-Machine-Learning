# Machine Learning Model Training and Evaluation

This repository contains code for training, evaluating, and comparing machine learning models on a dataset. The models included are Random Forest, Support Vector Machine (SVM), and Neural Network. The code is designed to be run in a Jupyter notebook, with each section separated into different cells for clarity and ease of use.

## Requirements

Before running the code, ensure you have the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Data

The code expects two CSV files:
- `Train_data.csv`: The training dataset.
- `Test_data.csv`: The test dataset.

Place these files in the same directory as your Jupyter notebook.

## Running the Code

1. **Imports**
    - Import the necessary libraries and set up the plotting styles.

2. **Reading the CSV Files**
    - Load the training and test datasets.
    - Display basic information about the datasets.

3. **Data Preprocessing**
    - Preprocess the data by encoding categorical variables and scaling numerical features.
    - Split the training data into training and validation sets.

4. **Feature Selection**
    - Select the top features based on ANOVA F-value.
    - Visualize the importance of the selected features.

5. **Model Training**
    - Train the following models on the selected features:
        - Random Forest
        - Support Vector Machine (SVM)
        - Neural Network
    - Each model is trained in a separate cell for clarity.

6. **Model Comparison**
    - Compare the models based on accuracy, precision, recall, and F1-score.
    - Visualize the performance of the models using confusion matrices and ROC curves.

7. **Feature Importance Analysis**
    - Analyze and visualize the feature importance for the Random Forest model.

8. **Predict on Test Data**
    - Preprocess the test data and make predictions using the best-performing model.
    - Save the predictions to CSV files and visualize the prediction distribution.

## Usage

To run the code, open the Jupyter notebook and execute each cell in order. The code will output various plots and results at each step, helping you understand the performance and behavior of the models.

### Example

Here is a brief overview of the steps you will follow:

1. **Start Jupyter Notebook**
    ```bash
    jupyter notebook
    ```

2. **Open the Notebook**

Navigate to the directory containing your notebook and open it.

3. **Execute Cells**

Run each cell in the notebook sequentially, starting from the imports and ending with the predictions on the test data.

## Visualizations

The notebook includes the following visualizations:

- Class distribution in the training data.
- Top features by ANOVA F-value.
- Confusion matrices for each model.
- ROC curves for model comparison.
- Feature importance for the Random Forest model.
- Prediction distribution for the test data.

These visualizations will help you gain insights into the data and the performance of the models.

## Output

The final output will include:

- A CSV file (`predictions.csv`) containing the predictions for the test data.
- A CSV file (`test_data_with_predictions.csv`) containing the original test data along with the predictions.
- Various PNG files of the plots generated during the analysis.

## Conclusion

In this project, we explored the training, evaluation, and comparison of different machine learning models for anomaly detection in a given dataset. The models evaluated included Random Forest, Support Vector Machine (SVM), and Neural Network.

After thorough preprocessing and feature selection, we trained each model on the selected features and evaluated their performance using various metrics such as accuracy, precision, recall, F1-score, and the Area Under the ROC Curve (AUC).

Among the models tested, the Random Forest classifier demonstrated outstanding performance with the highest accuracy and robust evaluation metrics across the board. Here are some key takeaways:

1. **Random Forest Classifier**:
    - Achieved the highest accuracy, indicating its effectiveness at distinguishing between normal and anomalous classes.
    - Displayed strong performance metrics including high precision, recall, and F1-score.
    - The ROC curve analysis further confirmed the Random Forest model's superior performance with a high AUC value, illustrating its excellent ability to differentiate between the classes.

2. **Support Vector Machine (SVM)**:
    - SVM also performed well, but slightly behind the Random Forest in terms of accuracy and other performance metrics.
    - Demonstrated good precision and recall but had a relatively lower AUC compared to Random Forest.

3. **Neural Network**:
    - The Neural Network model showed competitive performance but did not surpass the Random Forest in accuracy and other metrics.
    - While it had a higher computational cost, the Neural Network provided valuable insights into the data.

The feature importance analysis from the Random Forest model highlighted the most significant features contributing to the classifications, providing deeper insights into the underlying data patterns.

Based on these findings, the Random Forest classifier is recommended as the best-performing model for this anomaly detection task, balancing accuracy, interpretability, and computational efficiency.

Overall, this project underscores the importance of rigorous model evaluation and comparison to identify the most effective machine learning approach for a given problem. The Random Forest classifier emerged as a reliable and accurate choice, making it suitable for deployment in real-world applications where anomaly detection is critical.
