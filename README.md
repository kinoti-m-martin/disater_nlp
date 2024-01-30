## Disaster Text Detection System

![disaster nlp detection](https://github.com/kinoti-m-martin/disater_nlp/blob/main/Image-disasters-and-hazards.jpg)


The Disaster NLP Project aims to leverage Natural Language Processing (NLP) techniques to enhance the identification and classification of disaster-related information in textual data. The primary goal is to develop a robust model capable of accurately detecting and categorizing text as either disaster-related or not. This project can be instrumental in automating the analysis of vast amounts of information during emergencies, helping first responders, government agencies, and humanitarian organizations prioritize and respond effectively to crises.

### Data
- The link to the data source is [Kaggle](https://www.kaggle.com/competitions/nlp-getting-started/data)

- The data comprises of train.csv which has the following attributes: id, keyword, location, text, and target (classification of text); and test.csv whose attributes are:id, keyword, location, and text.

### Success Metrics:

- `Precision`: Precision measures the accuracy of the positive predictions made by the model. In the context of disaster NLP, it indicates the proportion of predicted disaster-related text that is actually relevant. A high precision score reflects the model's ability to avoid false positives.

- `Recall (Sensitivity)`: Recall measures the model's ability to capture all the relevant disaster-related text in the dataset. In this project, recall represents the proportion of actual disaster-related text that is correctly identified by the model. A high recall score indicates the model's effectiveness in avoiding false negatives.

- `F1 Score`: The F1 score is the harmonic mean of precision and recall. It provides a balanced assessment of the model's performance by considering both false positives and false negatives. A high F1 score indicates a well-rounded model that strikes a balance between precision and recall.

### Model
The best performing model for this task is the [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), which has been used in training and deployment.

### Deployment
[FastAPI](https://fastapi.tiangolo.com/) library has been used for this operation.