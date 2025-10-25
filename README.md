🚀 Sentimental Analysis Web App
A High-Performance Sentiment Predictor Built with Scientific Precision
This is the deployment repository for a custom-trained Machine Learning model designed to predict sentiment (Positive/Negative) from text input. The application is built on Streamlit for an elegant, interactive user interface.

✨ Project Highlights
📊 Model Performance & Dataset Scale

<img width="491" height="239" alt="image" src="https://github.com/user-attachments/assets/c94e78c0-0dbf-4f93-a4d9-1f470c5947e1" />


🛠️ The Power of PreprocessingThe high-performance score of 77.6% was achieved through an exhaustive and rigorous preprocessing pipeline, which is considered the cornerstone of this project.Data Normalization: The massive $1.6$ million tweet dataset was subjected to maximum normalization, including aggressive cleansing of noise (links, usernames, special characters) and meticulous stemming (using the Porter Stemmer) to reduce variance and capture the true semantic content of the language.Vector Consistency: The model ensures robust predictions by loading the pre-fitted TF-IDF Vectorizer (via pickle), guaranteeing that all new user inputs are mapped to the exact feature space learned from the 1.6 million original tweets.

⚙️ Repository Structure & Usage
This application is built entirely in Python and requires minimal setup to run:

1. Clone the repository.

2. Ensure Model Assets are Present: Verify that the trained model (trained_model.sav) and the fitted vectorizer (fitted_vectorizer.pkl) are present in the root directory.

3. Install Dependencies: Activate your environment and install the required packages:
   <img width="835" height="128" alt="image" src="https://github.com/user-attachments/assets/34f1c538-8695-4a18-bb3e-10845feda777" />

4. Launch the App: Run the Streamlit application from your terminal:
   <img width="825" height="123" alt="image" src="https://github.com/user-attachments/assets/85779c51-806b-4a58-861a-38ea6bd1479d" />
