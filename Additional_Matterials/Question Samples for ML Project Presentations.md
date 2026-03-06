# Question Samples for ML Project Presentations

---

## SECTION 1: Dataset Understanding & Exploration

- What is the size of your dataset (number of samples and features)?

- What type of problem does your dataset represent - classification or regression?

- Did you find any missing values in your dataset? How did you handle them?

- What is the distribution of classes in your dataset? Is it balanced or imbalanced?

- Did you identify any outliers in your data? What method did you use to detect them?

- What is the range of values for your target variable?

- Which feature in your dataset has the strongest correlation with the target variable?

- What data type are your features - numerical, categorical, or mixed?

- Did you create any visualizations during EDA? Which one was most informative?

- Did you encounter any duplicate records in your dataset?

---

## SECTION 2: Data Preprocessing & Feature Engineering

- What approach did you take to handle unscaled features? What method did you choose for normalization?

- Did you apply feature scaling to both training and test data? Which set did you fit the scaler on?

- How did you encode categorical variables in your dataset?

- Did you create any new features through feature engineering? What was the rationale?

- What percentage of your data did you use for training versus testing?

- Did you use stratified sampling when splitting your data? Why or why not?

- What is the random seed you used for reproducibility?

- Did you apply any data augmentation techniques? Which ones and why? Why not?

- How did you handle features with very different scales or ranges?

- Did you perform any dimensionality reduction? If yes, which technique?

---

## SECTION 3: Train/Validation/Test Splits

- What is the difference between validation set and test set?

- What split ratio did you use (e.g., 70/15/15 or 80/20)?

- Why is it important not to touch the test set during model development?

- What is cross-validation? Did you use cross-validation? If yes, how many folds?

- What is the purpose of using a validation set during training?

- Why should you fit the scaler only on training data and not on the entire dataset?

- Did you ensure that the class distribution is similar across train/validation/test sets?

- What would happen if you used the test set to tune hyperparameters?

- How does the size of your training set affect model performance?

---

## SECTION 4: Model Selection & Architecture

- Which model did you choose as your final model and why?

- How many different models did you compare before selecting the final one?

- What are the key differences between Model A and Model B (For Example CNN and NN)?

- Why might a simpler model be preferred over a complex one with similar performance?

- If you used a neural network, how many hidden layers did you include?

- What activation function did you use in hidden layers? Why?

- For CNN models: What is the purpose of the convolutional layer?

- For CNN models: Why do we use pooling layers?

- What is the difference between a fully connected layer and a convolutional layer?

- How did you determine the architecture of your neural network?

---

## SECTION 5: Training Process

- How many epochs did you train your model for?

- What batch size did you use during training? How did you choose it?

- Did your training loss decrease consistently, or did you observe fluctuations?

- What optimizer did you use (Adam, SGD, RMSprop) and why?

- How long did it take to train your model (approximately)?

- Did you monitor validation loss during training? What trend did you observe?

- At what epoch did your model achieve the best validation performance?

---

## SECTION 6: Loss Functions & Optimization

- What loss function did you use for your model?

- What does Mean Squared Error (MSE) measure in regression?

- What is the role of the learning rate in optimization?

- What happens if the learning rate is too high?

- What happens if the learning rate is too low?

---

## SECTION 7: Evaluation Metrics

- What is your model's test accuracy?

- Why is accuracy not always a good metric for imbalanced datasets?

- What is the difference between precision and recall?

- Can you explain what the F1-score represents?

- What is the confusion matrix for your model on the test set?

- For regression: What is your model's R² score? What does it mean?

- Which metric did you prioritize when evaluating your model and why?

- How would you interpret a precision of 0.85 in your problem context?

---

## SECTION 8: Hyperparameters

- What are hyperparameters and how do they differ from model parameters?

- Which hyperparameters did you tune in your model?

- For neural networks: How did you choose the number of neurons per layer?

- What is the difference between a hyperparameter and a learned weight?

- Did changing a specific hyperparameter significantly improve your results?

---

## SECTION 9: Overfitting & Underfitting

- What is overfitting and how can you detect it?

- Did your model show signs of overfitting? How do you know?

- What techniques did you use to prevent overfitting?

- If training accuracy is 99% but test accuracy is 70%, what is the problem?

- What is underfitting and when does it occur?

- How would you address overfitting if you detected it after training?

- What is the relationship between model complexity and overfitting?

---

## SECTION 10: Model Comparison & Deep Learning Theory

- How does your CNN compare to a fully connected neural network on the same data?

- Why are convolutional layers better than dense layers for image data?

- What is transfer learning and did you use it in your project?

- How does a pre-trained model like VGG16 or ResNet50 help with limited data?

- What is the purpose of a softmax activation in the output layer?

- What is backpropagation and why is it important for neural network training?

- If you had more time and resources, what would you do to improve your model?

- What real-world application or deployment scenario could your model be used for?

---
