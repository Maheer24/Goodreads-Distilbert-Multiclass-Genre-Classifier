import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AutoTokenizer
import torch
import re
from bs4 import BeautifulSoup

model_path = "maheer24/distilbert-genre-classification"

try:
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

st.title("Predict Genre From Summary 📖")

user_input = st.text_area("Enter your text here:",height=250)

if st.button("Predict"):
    if not user_input:
        st.warning("Please enter some text.")
    else:
        try:
            def clean_summary(summary):
                stopwords = {'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are',  'aren', "aren't", 'as', 'at',
                     'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both','but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn',
                     "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had',
                     'hadn', "hadn't",'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself',
                     'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', 
                    "mightn't",'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 
                    'once', 'only', 'or','other', 'our', 'ours', 'ourselves', 'out','over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she's", 
                     'should', "should've", 'shouldn',"shouldn't",'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 
                     'themselves', 'then', 'there', 'these', 'they','this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 
                     'wasn', "wasn't",'we', 'were', 'weren', "weren't", 'what', 'when','where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', 
                  "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", "you're", "you've",'your', 'yours', 'yourself', 'yourselves'}

                summary = re.sub(
                    r'(\bNew York Times Bestseller|Winner of the Pulitzer Prize|ANDNEW YORK TIMESBESTSELLERIn|ISBN|ASIN\b'
                    r'|Librarian note|Alternate cover edition.*?|(This|There) is an alternate Cover Edition.*?|forASIN:\s*\S+|'
                    r'\b\w*(ASIN|ISBN)\w*\b'
                    r'ASIN:\s*\S+|ISBN[:\d]+)',
                    '',
                    summary,
                    flags=re.IGNORECASE
                    ).strip()


                summary = re.sub(r'\b[Ww]{2}\b', 'world war', summary)

                # Remove characters like M.D. or m.d.
                summary = re.sub(r'[A-Za-z]\.[A-Za-z]\.', '', summary)

                # Remove characters like M. or m.
                summary = re.sub(r'\b[A-Za-z]\.(?=\s|$)','',summary)

                # Remove characters like DR. or dr.
                summary = re.sub(r'\b[A-Za-z]{2}\.\s?', '', summary)

                # Remove html tags
                summary = BeautifulSoup(summary, 'html.parser').get_text()

                # Remove special characters and punct(i.e. ?, @, ',' etc) (\. leaves full stop,\w leaves word characters like _,a,z , \s leaves white space)
                summary = re.sub(r'[^\w\s\.]', ' ', summary)

                # Remove underscores
                summary = re.sub(r'\_+',' ', summary)

                # Remove numbers
                summary = re.sub(r'[0-9]','',summary)

                # Remove Roman numerals
                summary = re.sub(r'\b[IVXLCDM]+\b','',summary)

                # Convert to lower case
                summary = summary.lower()

                # Replace ellipses with white space
                summary = re.sub(r'\.\.\.*', ' ', summary)
                summary = re.sub(r'\.\s*\.\s*\.+', ' ', summary)
                summary = re.sub(r'\s*\.\s*\.\s*\.',' ', summary)

                # Remove stop words
                summary = ' '.join([word for word in summary.split() if word not in stopwords])

                summary = re.sub(
                    r'award winning author|world bestselling authors|new york timesbestselling author|new york timesbestselleryou|new york times bestselling|goodreads|phenomenal worldwide bestseller|alternate cover|newer edition foundhere|national book award|pulitzer prize|national book award winner|pulitzer prize winner|new york timesandusa todaybestselling',
                    '',
                    summary,
                    flags = re.IGNORECASE
                ).strip()

                # Remove extra spaces and clean text
                # \s checks for white space, + matches for preceding characters(i.e. one or more whitespace)
                summary = re.sub(r'\s+', ' ', summary).strip()

                # Remove white space before full stop
                summary = re.sub(r'\s+(?=\.)', '', summary)

                # Ensure there is a single white space after each full stop.
                summary = re.sub(r'\.(?=\S)', '. ', summary)

                return summary
            pre_processed_input = clean_summary(user_input)
            inputs = tokenizer(pre_processed_input, padding = True, truncation = True, return_tensors = 'pt')

            with torch.no_grad():
                outputs = model(**inputs)

                predictions = torch.argmax(outputs.logits, dim=-1)

                class_names = [ "Biographies and Memoirs", "Classics and Historical", "Fantasy", "Fiction", 
                               "Horror and Paranormal", "Inspirational and Self-Help", "Literary Fiction", "Mystery","Psychology", 
                                "Romance", "Science Fiction", "Science and Technology", "Suspense and Thriller", 
                               "Youth Literature (Children and Young Adult)"]

                predicted_class_index = predictions.item()
                predicted_class = class_names[predicted_class_index]
                
            st.success(f"Predicted Genre: {predicted_class}")
        except Exception as e:
            st.error(f"An error occured: {e}")


