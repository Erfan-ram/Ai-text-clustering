import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import numpy as np
import requests
import nltk
from uuid import uuid4
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from random import choice

main_categories = {
    "hotnews": "اخبار داغ",
    "politics": "سیاست",
    "economy": "اقتصاد",
    "society": "اجتماعی",
    "international": "بین‌الملل",
    "sports": "ورزش",
    "culture": "فرهنگ",
    "accidents": "حوادث",
    "science": "علم",
    "photo": "عکس",
}

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def extract_site_caption():
    response = requests.get(
        url="https://khabaryab.net/xxxx",
        headers={'Authorization': 'xxxxx'}
    )

    if response.status_code == 200:
        data = response.json()
        # print(data)
        site_captions = []
        for eachnews in data['news']:
            if eachnews['summary'] != "":
                site_captions.append({
                    'caption': eachnews['summary'],
                    'source': 'siteCaptions',
                    'id': uuid4().hex,
                }) 
        
        # with open('siteCaptions.json', 'w', encoding='utf-8') as file:
        #     json.dump(site_captions, file, indent=4, ensure_ascii=False)
            
        if len(site_captions)<5:
            return None
        return site_captions

def extract_scraper_captions(data : dict):
    captions = []
    for category in main_categories.keys():
        # print(category)
        for article in data[category]:
            if 'Caption' in article and article['Caption'] != "":
                captions.append({
                    'caption': article['Caption'],
                    'source': 'scraperCaptions',
                    'id': article['uid']
                })
    # with open('scraperCaptions.json', 'w', encoding='utf-8') as file:
    #     json.dump(captions, file, indent=4, ensure_ascii=False)

    print(f"Extracted {len(captions)} captions")
    if len(captions) < 5:
        return None
    return captions

# A simple function to clean up text
def preprocess_text(text):
    # Get rid of special characters and numbers
    text = re.sub(r'[^ا-یA-Za-z\s]', '', text)  # Let Persian characters through
    # Break the text into words
    words = text.lower().split()
    # Drop stop words and lemmatize the rest
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Function to read and preprocess captions from a file
def load_and_preprocess_captions(file_name):
    captions = []
    for item in file_name:
        captions.append((item['source'], item['id'], item['caption'], preprocess_text(item['caption'])))
    return captions

def rundbscan(newsdict : dict):
    # Download some NLTK stuff we need
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    # Load and preprocess captions from both files
    sitecap = extract_site_caption()
    scrapcap = extract_scraper_captions(newsdict)
    
    if not sitecap and not scrapcap:
        print("failed to get caption from scraper or site ")
        return
    captions_file1 = load_and_preprocess_captions(scrapcap)
    captions_file2 = load_and_preprocess_captions(sitecap)

    # Combine captions from both files
    all_captions = captions_file1 + captions_file2

    # Convert to DataFrame for easy processing
    df = pd.DataFrame(all_captions, columns=['Source', 'ID', 'Caption', 'Processed_Caption'])

    # Set up the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Convert the cleaned captions into TF-IDF vectors
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Processed_Caption'])

    # Calculate the cosine similarity between all captions
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Convert cosine similarity to distance (non-negative)
    distance_matrix = 1 - cosine_sim  # Distance in range [0, 2]
    distance_matrix = np.clip(distance_matrix, 0, 2)  # Ensure no negative values

    # Apply DBSCAN for clustering based on cosine similarity
    eps_value = 0.65  # This is the similarity threshold (adjustable)
    min_samples_value = 2  # Minimum number of captions in a group

    # Apply DBSCAN clustering on the distance matrix
    clustering = DBSCAN(eps=eps_value, min_samples=min_samples_value, metric='precomputed')
    labels = clustering.fit_predict(distance_matrix)

    # Group the captions based on their cluster labels
    grouped_captions = {}
    for label in np.unique(labels):
        if label != -1:  # Exclude noise (-1) if present
            grouped_captions[str(int(label))] = []  # Convert label to string
            group_df = df[labels == label]
            for _, row in group_df.iterrows():
                grouped_captions[str(int(label))].append({
                    'Caption': row['Caption'],
                    'ID': row['ID'],
                    'Source': row['Source'],
                })

    # site_captions_group_count = sum(
    #     any(caption['Source'] == 'siteCaptions' for caption in captions)
    #     for captions in grouped_captions.values()
    # )

    # print(f"Total number of groups: {len(grouped_captions)}")
    # print(f"Number of groups with at least one 'siteCaptions': {site_captions_group_count}")

    filtered_grouped_captions = {
        label: captions for label, captions in grouped_captions.items()
        if not any(caption['Source'] == 'siteCaptions' for caption in captions)
    }
    
    final_captions = {}
    # Randomly filter out one caption from each group
    for label, captions in filtered_grouped_captions.items():
        if len(captions) > 1:
            random_caption = choice(captions)
            final_captions[label] = [random_caption]
    
    
    # Find UIDs of news that are in grouped_captions but not in final_captions
    repeated_news_uids = []
    all_caption_ids = []
    final_caption_ids = []
    
    for label, captions in final_captions.items():
        for eachnews in captions:
            final_caption_ids.append(eachnews['ID'])
            
    for label, captions in grouped_captions.items():
        for eachnews in captions:
            # pass news in database
            if eachnews['Source'] == 'siteCaptions':
                continue
            all_caption_ids.append(eachnews['ID'])
            
    print (f"all cap: {len(all_caption_ids)},final unique: {len(final_caption_ids)}")
    
    for caption_id in all_caption_ids:
        if caption_id not in final_caption_ids:
            repeated_news_uids.append(caption_id)

    # with open('repeated_news_uids.json', 'w', encoding='utf-8') as json_file:
    #     json.dump(repeated_news_uids, json_file, ensure_ascii=False, indent=4)

    print(f"Repeated news UIDs saved to repeated_news_uids.json {len(repeated_news_uids)}")

    # with open('similar_groups.json', 'w', encoding='utf-8') as json_file:
    #     json.dump(grouped_captions, json_file, ensure_ascii=False, indent=4)
    # with open('filterd_similar_groups.json', 'w', encoding='utf-8') as json_file:
    #     json.dump(filtered_grouped_captions, json_file, ensure_ascii=False, indent=4)
    # with open('final_captions.json', 'w', encoding='utf-8') as json_file:
    #     json.dump(final_captions, json_file, ensure_ascii=False, indent=4)

    # print("Grouped captions saved to similar_groups.json")
    
    return repeated_news_uids
    
# extract_site_caption()

# if __name__ == '__main__':
#     # main()
#     mine()