# Goodreads-Distilbert-Multiclass-Genre-Classifier

![streamlit_app](https://github.com/user-attachments/assets/0c35e1fe-230c-483c-997d-d4b6f56e42b0)

## **a. Introduction**

This project focuses on building a solution to classify book genres based on summaries scraped from GoodReads. Using a combination of web scraping, data preprocessing, and fine-tuning DistilBERT—a transformer-based language model—for the specific task of multi-class classification, the project achieves effective results. The final solution is deployed as a user-friendly Streamlit app.

## **b. Objective**

The primary objective of this project is to develop a robust model capable of predicting book genres from textual summaries while addressing challenges like class imbalance and noisy data.

## **c. Key Steps in Workflow**

### **Data Collection**

Scraped 7,500 book summaries and their corresponding genres from Goodreads using BeautifulSoup.

### **Data Preprocessing**

#### **1. Cleaning Data**

- Dropped unnecessary columns.

- Removed null values and duplicates.

- Filtered out non-English summaries.

#### **2. Genre Reduction and Grouping**

- Reduced the list of genres per book to a single representative genre.

- Removed unnecessary or underrepresented genres to reduce noise and class imbalance.

- Merged similar genres using a predefined dictionary for mapping.

#### **3. Addressing Class Imbalance**

- Augmented the dataset by merging it with a Kaggle dataset to balance genre representation.

#### **4. Encoding and Splitting**

- Encoded genres for model compatibility.

- Performed a train-test split with stratification to preserve class distribution.

#### **5. Handling Imbalance in Training**

- Computed class weights for training.

- Incorporated focal loss to mitigate the effects of class imbalance.

### **Model Development**

Fine-tuned a pre-trained DistilBERT model for multi-class classification of genres.

### **Deployment**

Deployed the fine-tuned model on Streamlit, making it accessible for predictions through an interactive web interface.

## **d. Tools and Libraries Used**

All the libraries along with their versions are listed [here](https://github.com/Maheer24/Goodreads-Distilbert-Multiclass-Genre-Classifier/blob/main/requirements.txt)

## **e. Streamlit App**

The Streamlit app provides an easy-to-use interface for users to predict the genre of a book based on its summary. Users can input a book summary, and the app leverages the fine-tuned DistilBERT model to output the most probable genre.
