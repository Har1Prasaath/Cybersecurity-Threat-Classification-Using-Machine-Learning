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
pip install pandas numpy matplotlib seaborn scikit-learn

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
This repository provides a comprehensive framework for training, evaluating, and comparing machine learning models. By following the steps outlined in the notebook, you can gain valuable insights into your data and the performance of different models.