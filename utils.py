import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import auc
import matplotlib.pyplot as plt

######## Data Loading Functions ########

def xml_to_df(path_to_xml):
    """Loads an XML file from Stack Exchange as a pandas dataframe."""
    tree = ET.parse(path_to_xml)
    root = tree.getroot()
    children = [child.attrib for child in root]
    return pd.DataFrame(children)

######## Feature Extraction Functions ########

def build_features(use_tags=False, use_nlp=None):
    """Returns a dataframe of features and targets for predicting whether or not a question will have 
    an answer. Optional argument `use_nlp` should be a dictionary with keys `vectorizer` and 
    `text_col`."""
    df = load_questions()
    y = build_target(df)
    features = build_simple_features(df)
    if use_tags:
        features = pd.concat([features, build_tag_features(df)], axis=1)
    if use_nlp:
        features = pd.concat([features, build_nlp_features(df, vectorizer=use_nlp["vectorizer"],
                                                           text_col=use_nlp["text_col"])
                             ],axis=1)
    return features, y
    
def load_questions():
    """Returns a dataframe of questions before 2021-02-01 from Stack Exchange."""
    df = xml_to_df("data/gardening.stackexchange.com/Posts.xml")
    df = df[(df.PostTypeId == "1") & (df.CreationDate < "2021-02-01")].reset_index(drop=True)
    return df

def build_target(df, verbose=True):
    """Creates target varible from df."""
    df["HasAnswer"] = (df.AnswerCount.astype(int) > 0).astype(int)  # Target
    if verbose:
        print(df[["HasAnswer"]].describe())
    return df[["HasAnswer"]]

simple_features = ["ContainsImage", "TagCount", "Untagged", "TitleLength", "BodyLength", 
                       "IsTitleQuestion", "BodyQuestionCount", "ViewCount"]

def build_simple_features(df, verbose=True):
    """Creates hand-made / actionable features from df."""
    df["ContainsImage"] = df.Body.str.contains("img src=").astype(int)
    df["TagCount"] = df.Tags.apply(lambda x: 0 if "untagged" in x else str(x).count("<"))
    df["Untagged"] = (df.TagCount == 0).astype(int)
    df["TitleLength"] = df.Title.apply(lambda x: len(word_tokenize(x)))
    df["BodyLength"] = df.Body.apply(lambda x: len(word_tokenize(x)))
    df["IsTitleQuestion"] = df.Title.apply(lambda x: 1 if "?" in x else 0)
    df["BodyQuestionCount"] = df.Body.apply(lambda x: x.count("?"))
    df["ViewCount"] = df.ViewCount.astype(int)
    if verbose:
        print(df[simple_features].describe(include="all"))
    return df[simple_features]

def build_tag_features(df):
    """Creates indicator columns for tags from df."""
    tags = xml_to_df("data/gardening.stackexchange.com/Tags.xml").TagName.tolist()
    for tag in tags:
        df[tag] = df.Tags.str.contains(tag).astype(int)
    return df[tags]

def build_nlp_features(df, vectorizer, text_col):
    """Creates NLP features from df."""
    if vectorizer == "count":
        vectorizer = CountVectorizer(stop_words="english", max_features=1000)
    elif vectorizer == "tfidf":
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X = vectorizer.fit_transform(df[text_col].tolist())
    X = pd.DataFrame(columns=vectorizer.get_feature_names(), data=X.toarray())
    return X

######## Evaluation Functions ########

def evaluate(model, X_test, y_test):
    """Plots AUC for a model and test set."""
    y_score = pd.DataFrame(model.predict(X_test), columns=["y_score"]).reset_index(drop=True)
    y = pd.concat([y_test.reset_index(drop=True), y_score], axis=1)
    y.columns = ["y_test", "y_score"]
    y = y.sort_values("y_score", ascending=True) \
            .reset_index(drop=True)
    y["precision_0class"] = (y.y_test == 0).astype(int).cumsum() / (y.index + 1)
    y["recall_0class"] = (y.y_test == 0).astype(int).cumsum() / (len(y) - y.y_test.sum())
    auc_0class = auc(x=y.recall_0class, y=y.precision_0class)
    plt.plot(y.recall_0class, y.precision_0class)
    plt.title(f"Test set evaluation\nPrecision-recall curve for no-answer class\nAUC={auc_0class:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

def feature_importances(model, feature_names, num_features=20):
    """Plot top feature importances for a gradient boosted tree model."""
    importances = model.feature_importances_[:num_features]
    indices = np.argsort(importances)[:num_features]
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
