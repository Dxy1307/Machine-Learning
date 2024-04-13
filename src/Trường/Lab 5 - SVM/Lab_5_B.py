# Spam Classification using SVM
# Preprocessing Emails
import re, string
from stemming.porter2 import stem

def processEmail(text):
    text = text.lower()                                         # lower case
    text = re.sub("<[^<>]+>", " ", text)                        # strip all html
    text = re.sub("[0-9]+", "number", text)                     # handle numbers
    text = re.sub("(http|https)://[^\s]*", "httpaddr", text)    # handle URLS
    text = re.sub("[^\s]+@[^\s]+", "emailaddr", text)           # handle email address
    text = re.sub("[$]+", "dollar", text)                       # handle $ sign
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove any punctuation
    text = re.sub("\s+", " ", text).strip()                     # replace multi white space with one space
    text = re.sub("[^a-zA-Z0-9 ]", "", text)                    # remove any other non-alphanumeric characters
    text = " ".join([stem(word) for word in text.split(" ")])   # stemming all words
    text = " ".join([word for word in text.split(" ") if len(word) > 1]) # removing too short words
    return text

sample_email = """
> Anyone knows how much it costs to host a web portal ?
>
Well, it depends on how many  visitors youre   expecting. This can be
anywhere from less than 10 bucks a month to a couple of $100. You
should checkout http://www.rackspace.com/ or perhaps Amazon EC2 if
youre running something big..
<img src="haha.png" />
To unsubscribe yourself from this mailing list, send an email to:
groupname-unsubscribe@egroups.com
"""

preprocessed_email = processEmail(sample_email)
print("***** Preprocessed email: *****\n{}".format(preprocessed_email))

# Extracting features from emails
import numpy as np

# This loads a vocabulary array of 1899 words
vocabulary = np.array([word.strip() for word in open("datasets/vocab.txt")])

def email_to_vector(vocabulary, original_email):
    email_words = processEmail(original_email).split(" ") # list of words in preprocessed email
    set_words = set(email_words)
    return np.array([1 if i in set_words else 0 for i in vocabulary])

x = email_to_vector(vocabulary, sample_email)
print(x)
print("The number of 1 in x:", np.sum(x==1))

# training SVM for spam classification
import matplotlib.pylab as plt
from scipy.io import loadmat

mat = loadmat("datasets/spamTrain.mat")
X = mat["X"]
y = mat["y"].reshape(len(X))
print("X.shape:", X.shape, "y.shape", y.shape)

mat = loadmat("datasets/spamTest.mat")
Xtest = mat["Xtest"]
ytest = mat["ytest"].reshape(len(Xtest))
print("Xtest.shape:", Xtest.shape, "ytest.shape", ytest.shape)

from sklearn.decomposition import PCA
X_pca = PCA(n_components=2).fit_transform(X)

fig, ax = plt.subplots()
X0_pca, X1_pca = X_pca[y==0], X_pca[y==1]
ax.scatter(X0_pca[:,0], X0_pca[:,1], color='green', marker='.', label='Non-spam (y=0)')
ax.scatter(X1_pca[:,0], X1_pca[:,1], color='red', marker='1', label='Spam (y=1)')
ax.set_xlabel("First principal component of PCA")
ax.set_ylabel("Second principal component of PCA")
ax.set_title("Training dataset visualization using PCA")
ax.legend()
# plt.show()

from sklearn.svm import SVC
clf = SVC(C=0.03, kernel='linear').fit(X, y)

acc_train = np.mean(clf.predict(X) == y) * 100
acc_test = np.mean(clf.predict(Xtest) == ytest) * 100
print("Training Accuracy = {}, Test Accuracy = {}".format(acc_train, acc_test))

# Top predictors for spam
theta = np.concatenate((clf.intercept_, clf.coef_[0]))
theta = theta[1:]

indices = np.argsort(theta) # ascending sort
indices = indices[-10:]     # top 10 highest parameters indices
print(vocabulary[indices])

# try your own emails
email_text = """
Master’s degree in Cyber Security

Are you interested in earning a Master’s degree in Cyber Security?

Study 100% online with the University of London, with a programme designed to help you stay ahead in the rapidly changing field of cyber security and progress your career with CISO-ready skills.

Why choose this programme?

Flexibility: Work through the programme in your own time studying up to 2 modules per term. You have 2 to 4 years to complete all modules.
Funding: Take advantage of the pay-as-you-go fee structure meaning you’ll only be charged for the modules you choose to take.
Knowledge: Learn how to protect the fabric of society through a comprehensive set of modules, including Applied Cryptography, and Security and Behaviour Change.
Hands-on learning: Understand how to connect academic and theoretical cyber security knowledge with hands-on lab sessions simulating real-life scenarios to gain practical experience.
Faculty: Gain insights from pioneering and influential researchers, academics, and professionals from Royal Holloway’s Information Security Group.

"""

x = email_to_vector(vocabulary, email_text)
pred = clf.predict([x])[0]
print("Prediction:", "SPAM" if pred==1 else "NOT SPAM")